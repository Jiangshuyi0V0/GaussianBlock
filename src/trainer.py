import argparse
from pathlib import Path
import time
from toolz import merge, keyfilter, valmap
import warnings

import numpy as np
from pytorch3d.structures import Meshes
import torch
from torchvision.transforms import functional as F

from dataset import create_train_val_test_loader, create_single_loader
from model import create_model
from optimizer import create_optimizer
from scheduler import create_scheduler
from utils import use_seed, path_exists, path_mkdir, load_yaml, dump_yaml
from utils.dtu_eval import evaluate_mesh
from utils.image import ImageLogger
from utils.logger import create_logger, print_log, print_warning
from utils.metrics import Metrics
from utils.path import CONFIGS_PATH, RUNS_PATH, DATASETS_PATH
from utils.plot import plot_lines, Visualizer, get_fancy_cmap
from utils.pytorch import get_torch_device, torch_to

from timm.utils import ModelEmaV2

LIGHT_MEMORY_RESULTS = True
LOG_FMT = 'Epoch [{}/{}], Iter [{}/{}], {}'.format
LOG_ITER_FMT = '{}'.format
N_VIZ_SAMPLES = 4
torch.backends.cudnn.benchmark = True  # XXX accelerate training if fixed input size for each layer
warnings.filterwarnings('ignore')




class Trainer:
    """Pipeline to train a model on a particular dataset, both specified by a config cfg."""

    @use_seed()
    def __init__(self, cfg, run_dir, gpu=None, rank=None, world_size=None):
        self.run_dir = path_mkdir(run_dir)
        self.device = get_torch_device(gpu, verbose=True)
        self.train_loader, self.val_loader, self.test_loader = create_train_val_test_loader(cfg)
        self.colored_loader = create_single_loader(cfg)
        self.model = create_model(cfg, self.train_loader.dataset.img_size).to(self.device)
        self.ema = ModelEmaV2(self.model, decay=0.9, device=self.device)  # init ema
        self.ema.module.to(self.device)
        self.cfg = cfg
        self.optimizer = create_optimizer(cfg, self.model)
        self.scheduler = create_scheduler(cfg, self.optimizer)
        self.epoch_start, self.batch_start = 1, 1
        self.n_epoches, self.n_batches = cfg['training'].get('n_epoches'), len(self.train_loader)
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.load_from(cfg)
        self.scheduler = create_scheduler(cfg, self.optimizer)
        # self.cur_lr = 5e-5
        print_log(f'Training state: epoch={self.epoch_start}, batch={self.batch_start}, lr={self.cur_lr}')

        # Logging metrics
        append = self.epoch_start > 1
        self.train_stat_interval = cfg['training']['train_stat_interval']
        self.val_stat_interval = cfg['training']['val_stat_interval']
        self.save_epoches = cfg['training'].get('save_epoches', [])
        names = self.model.loss_names if hasattr(self.model, 'loss_names') else ['loss']
        self.train_metrics = Metrics(*['time/img'] + names, log_file=self.run_dir / 'train_metrics.tsv', append=append)
        names = [f'alpha{k}' for k in range(self.model.n_blocks)]
        self.val_metrics = Metrics(*names, log_file=self.run_dir / 'val_metrics.tsv', append=append)
        names = [f'block{k}' for k in range(self.model.n_blocks)]
        self.block_metrics = Metrics(*['time/img'] + names, log_file=self.run_dir / 'train_blk_metrics.tsv', append=append)

        # Logging visuals
        with use_seed(12345):
            samples, labels = next(iter(self.val_loader if len(self.val_loader) > 0 else self.train_loader))
        self.viz_samples = valmap(lambda t: t.to(self.device)[:N_VIZ_SAMPLES], samples)
        self.viz_labels = valmap(lambda t: t.to(self.device)[:N_VIZ_SAMPLES], labels)
        out_ext = 'jpg' if LIGHT_MEMORY_RESULTS else 'png'
        self.rec_logger = ImageLogger(self.run_dir / 'reconstructions', self.viz_samples, out_ext=out_ext)
        self.rec2_logger = ImageLogger(self.run_dir / 'reconstructions_hard', self.viz_samples, out_ext=out_ext)
        self.rec3_logger = ImageLogger(self.run_dir / 'reconstructions_syn', self.viz_samples, out_ext='png')
        self.txt_logger = ImageLogger(self.run_dir / 'txt_blocks', out_ext=out_ext)
        if self.with_training:
            viz_port = cfg['training'].get('visualizer_port')
            self.visualizer = Visualizer(None, self.run_dir)
        else:  # no visualizer if eval only
            self.visualizer = Visualizer(None, self.run_dir)

        # logging Exp settings
        self.if_bin_render = cfg['Exp_setting']['Exp_setting']['if_bin_render']
        self.blk_loss_metrics = None

    @property
    def with_training(self):
        return self.epoch_start < self.n_epoches

    @property
    def dataset(self):
        return self.train_loader.dataset

    def load_from(self, cfg):
        pretrained, resume = cfg['training'].get('pretrained'), cfg['training'].get('resume')
        assert not (pretrained is not None and resume is not None)
        tag = pretrained or resume
        if tag is not None:
            path = path_exists(self.run_dir / pretrained)
            checkpoint = torch.load(path, map_location=self.device)
            self.model = create_model(cfg, self.train_loader.dataset.img_size).to(self.device)
            self.ema = ModelEmaV2(self.model, decay=0.9, device=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer = create_optimizer(cfg, self.model)
            if resume is not None:
                if checkpoint['batch'] == self.n_batches:
                    self.epoch_start, self.batch_start = checkpoint['epoch'] + 1, 1
                else:
                    self.epoch_start, self.batch_start = checkpoint['epoch'], checkpoint['batch'] + 1
                self.model.set_cur_epoch(checkpoint['epoch'])
                print_log(f'epoch_start={self.epoch_start}, batch_start={self.batch_start}')
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                except ValueError:
                    print_warning("ValueError: loaded optim state contains parameters that don't match")
                scheduler_state = keyfilter(lambda k: k in ['last_epoch', '_step_count', '_last_lr'], checkpoint['scheduler_state'])
                self.scheduler.load_state_dict(scheduler_state)
                self.cur_lr = self.scheduler.get_last_lr()[0]
                print_log(f'scheduler state_dict: {self.scheduler.state_dict()}')
            print_log(f'Checkpoint {path} loaded')

    def run(self, Exp_cfg):
        cur_iter = (self.epoch_start - 1) * self.n_batches + self.batch_start
        warmup_iter = Exp_cfg['warmup_iter']
        split_interval = Exp_cfg['split_interval']
        combine_interval = Exp_cfg['combine_interval']
        stop_split_iter = (self.n_epoches * self.n_batches) - Exp_cfg['stop_split']
        blk_backward_start_iter = Exp_cfg['blk_backward_start']
        for epoch in range(self.epoch_start, self.n_epoches + 1):
            batch_start = self.batch_start if epoch == self.epoch_start else 1
            for batch, ((images, labels), (col_images, col_labels)) in enumerate(zip(self.train_loader, self.colored_loader), start=1):
                if batch < batch_start:
                    continue
                warmup = True if cur_iter < warmup_iter else False
                blk_backward_start = False if cur_iter < blk_backward_start_iter else True
                is_split = True if cur_iter % split_interval == 0 and cur_iter < stop_split_iter else False
                is_combine = True if cur_iter % combine_interval == 0 else False
                print(f"blk_backward_start:{blk_backward_start}, blk_backward_start:{blk_backward_start_iter}, cur_iter:{cur_iter}, is_split:{is_split}, is_combine:{is_combine}")

                self.run_single_batch_train(images, labels, col_images, col_labels, warmup, Exp_cfg, cur_iter=cur_iter, split_blk=is_split, blk_backward_start=blk_backward_start, combine_blk=is_combine)
                if cur_iter % self.train_stat_interval == 0:
                    self.log_train_metrics(cur_iter, epoch, batch, warmup, split_blk=is_split, blk_backward_start=blk_backward_start, combine_blk=is_combine)

                if cur_iter % self.val_stat_interval == 0:
                    self.run_val_and_log(cur_iter, epoch, batch)
                    self.log_visualizations(cur_iter, Exp_cfg)
                    self.save(epoch=epoch, batch=batch)
                cur_iter += 1
            self.step(epoch + 1, batch=1)
            if epoch % 50 == 0 or epoch in self.save_epoches:
                self.save(epoch=epoch, batch=batch, checkpoint=True)

        N, B = (self.n_epoches, self.n_batches) if self.with_training else (self.epoch_start, self.batch_start)
        self.save(epoch=N, batch=B)
        self.save_metric_plots()
        self.evaluate()
        print_log('Training over')

    def run_single_batch_train(self, images, labels, col_images, col_labels, warmup, Exp_cfg, cur_iter, split_blk=False, blk_backward_start=False, combine_blk=False):
        start_time = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        B = len(images['imgs'])
        loss = self.model(torch_to(images, self.device), torch_to(labels, self.device))
        loss['total'].backward()
        dict_loss = {f'loss_{k}': v.detach().mean().item() for k, v in loss.items()}
        self.train_metrics.update(merge({'time/img': (time.time() - start_time) / B}, dict_loss), N=B)
        self.optimizer.step()
        self.ema_update()
        if not warmup:
            blk_loss_tot = torch.zeros(1).to(self.device)
            if combine_blk:
                N = self.model.n_blocks
                print_log(f'***Detecting combined block pairs***')
                blk_loss, combine_pair = self.model.compute_blk_losses(torch_to(col_images, self.device), Exp_cfg, combine_blk=combine_blk)
                self.optimizer_update(N)
                self.ema_update()
                if combine_pair is not None and N != self.model.n_blocks:
                    print_log(f'Before Single Optimization, loss -- {loss["total"]}')
                    self.log_visualizations(cur_iter, Exp_cfg)
                    for i in range(50):
                        self.run_single_blk_training(images, labels, col_images, Exp_cfg, num=1)
                    self.log_visualizations(cur_iter + 1, Exp_cfg)

            if split_blk:
                N = self.model.n_blocks
                print_log(f'***Detecting Splitting -- from {N} blocks***')
                blk_loss = self.model.compute_blk_losses(torch_to(col_images, self.device), Exp_cfg, split_blk=split_blk)
                print_log(f'***Splitting Over -- currently {self.model.n_blocks} blocks***')
                self.optimizer_update(N)
                self.ema_update()
                if N != self.model.n_blocks:
                    print_log(f'Before Single Optimization, loss -- {loss["total"]}, blk_loss -- {blk_loss}')
                    self.log_visualizations(cur_iter, Exp_cfg)
                    for i in range(50):
                        self.run_single_blk_training(images, labels, col_images, Exp_cfg, num=self.model.n_blocks-N)
                    self.log_visualizations(cur_iter + 1, Exp_cfg)

            if blk_backward_start:
                self.optimizer.zero_grad()
                blk_loss = self.model.compute_blk_losses(torch_to(col_images, self.device), Exp_cfg)
                for retain_index, blk_idx in enumerate(blk_loss):
                    retain = (retain_index < len(blk_loss) - 1)
                    try:
                        blk_loss[blk_idx].backward(retain_graph=retain)
                        blk_loss_tot += blk_loss[blk_idx]
                    except:
                        continue
                self.optimizer.step()
                self.ema_update()

            if combine_blk or split_blk or blk_backward_start:
                named_values = [(f'block{k}', 0 if a == -1 else a.item()) for k, a in blk_loss.items()]
                names = [f'block{k}' for k in range(self.model.n_blocks)]
                self.block_metrics.update(merge({'time/img': (time.time() - start_time) / B}, named_values), N=B, new_names=names)
                print_log(blk_loss_tot)


    def run_single_blk_training(self, images, labels, col_images, Exp_cfg, num=1):
        self.optimizer.zero_grad()
        loss = self.model(torch_to(images, self.device), torch_to(labels, self.device))
        loss_tot = loss['total']
        if num > 1:
            blk_loss = self.model.compute_blk_losses(torch_to(col_images, self.device), Exp_cfg)
            print_log(f'blk_loss -- {blk_loss}')
            for idx, blk_idx in enumerate(blk_loss):
                if idx < len(blk_loss) - num:
                    continue
                else:
                    try:
                        loss_tot += blk_loss[blk_idx]
                    except:
                        continue
        print_log(f'loss_tot -- {loss_tot}')
        loss_tot.backward()

        self.model.S.grad[0:-num] = 0.
        if self.model.alpha_logit.grad is not None:
            self.model.alpha_logit.grad[0:-num] = 0.
        self.model.R_6d.grad[0:-num] = 0.
        self.model.T.grad[0:-num] = 0.
        self.model.sq_eps.grad[0:-num] = 0.
        self.optimizer.zero_grad()
        loss = self.model(torch_to(images, self.device), torch_to(labels, self.device))
        loss_tot = loss['total']
        print_log(f'loss_tot -- {loss_tot}')
        loss_tot.backward()
        self.optimizer.step()
        self.ema_update()


    def ema_update(self):
        try:
            self.ema.update(self.model)
        except RuntimeError:
            for name, param in self.model.named_parameters():
                ema_param = getattr(self.ema.module, name)
                if ema_param.data.shape != param.data.shape:
                    # preserve existing values and only adjust the new dimensions.
                    new_ema_param = torch.zeros_like(param.data)

                    # Copy the existing values to the new EMA parameter
                    min_shape = [min(s1, s2) for s1, s2 in zip(ema_param.data.shape, param.data.shape)]
                    slices = tuple(slice(0, min_s) for min_s in min_shape)
                    new_ema_param[slices] = ema_param.data[slices]

                    # Update the EMA model's parameter
                    setattr(self.ema.module, name, torch.nn.Parameter(new_ema_param))
                    ema_param = new_ema_param
                # Update the EMA parameter with the current model's parameter
                ema_param.data.mul_(self.ema.decay).add_(param.data, alpha=1 - self.ema.decay)
            self.ema.module.n_blocks = self.model.n_blocks
            self.ema.module.blocks = self.model.blocks
            self.ema.module.sq_eta = self.model.sq_eta
            self.ema.module.sq_omega = self.model.sq_omega
            self.ema.module.textures = self.model.textures

    def optimizer_update(self, N=10):
        if N == self.model.n_blocks:
            return
        else:
            assert N < self.model.n_blocks
            self.optimizer = create_optimizer(self.cfg, self.model)


    @torch.no_grad()
    def run_val_and_log(self, it, epoch, batch):
        metrics = self.val_metrics
        opacities = self.ema.module.get_opacities()
        if (opacities > 0.01).sum() == 0:
            raise RuntimeError('No more blocks....')
        named_values = [(f'alpha{k}', a.item()) for k, a in enumerate(opacities)]
        metrics.update(dict(named_values), new_names = list(dict(named_values).keys()))
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'val_metrics: {metrics}')[:1000])
        cmap = get_fancy_cmap()
        colors = (cmap(np.linspace(0, 1, len(named_values) + 1)[1:]) * 255).astype(np.uint8)
        self.visualizer.upload_lineplot(it, metrics.get_named_values(), title='opacities', colors=colors)
        metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def step(self, epoch, batch):
        self.scheduler.step()
        self.model.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'LR update: lr={lr}'))

    def log_train_metrics(self, it, epoch, batch, warmup=False, split_blk=False, blk_backward_start=False, combine_blk=False):
        metrics = self.train_metrics
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'train_metrics: {metrics}')[:1000])
        self.visualizer.upload_lineplot(it, metrics.get_named_values(lambda s: 'loss' in s), title='train_losses')
        metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

        if not warmup and (split_blk or blk_backward_start or combine_blk):
            # named_values = [(f'block{k}', 0 if a == -1 else a.item()) for k, a in blk_loss.items()]
            cmap = get_fancy_cmap()
            colors = (cmap(np.linspace(0, 1, self.model.n_blocks + 1)[1:]) * 255).astype(np.uint8)
            block_metrics = self.block_metrics
            print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'train_block_metrics: {block_metrics}')[:1000])
            self.visualizer.upload_lineplot(it, block_metrics.get_named_values(lambda s: 'block' in s), title='block_losses', colors=colors)
            block_metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    @torch.no_grad()
    def log_visualizations(self, cur_iter, Exp_cfg):
        self.model.eval()
        # Log soft reconstructions with edges
        rec = self.ema.module.predict(self.viz_samples, self.viz_labels, w_edges=True)
        self.rec_logger.save(rec, cur_iter)
        imgs = F.resize(self.viz_samples['imgs'], rec.shape[-2:], antialias=True)
        self.visualizer.upload_images(torch.stack([imgs, rec], dim=1).reshape(-1, *imgs.shape[1:]), 'recons', 2)

        # Log hard reconstructions
        rec = self.ema.module.predict(self.viz_samples, self.viz_labels, filter_transparent=True)
        self.rec2_logger.save(rec, cur_iter)
        self.visualizer.upload_images(torch.stack([imgs, rec], dim=1).reshape(-1, *imgs.shape[1:]), 'recons_hard', 2)

        # Log rendering with synthetic colors
        rec = self.ema.module.predict_synthetic(self.viz_samples, self.viz_labels)
        self.rec3_logger.save(rec, cur_iter)
        self.visualizer.upload_images(torch.stack([imgs, rec], dim=1).reshape(-1, *imgs.shape[1:]), 'recons_syn', 2)

        if not self.if_bin_render:
            # Log textures
            txt = self.model.get_arranged_block_txt()
            self.txt_logger.save(txt, cur_iter)
            self.visualizer.upload_images(txt, 'textures', 1, max_size=256)

    @torch.no_grad()
    def log_debug_visualizations(self, cur_iter, Exp_cfg):
        self.model.eval()
        # Log soft reconstructions with edges
        rec, verts2d = self.ema.module.predict(self.viz_samples, self.viz_labels, w_edges=True, w_vert=True,
                                               separate_blocks=True)
        self.rec_logger.save(rec, cur_iter, separate_blocks=True)

    def save(self, epoch, batch, checkpoint=False):
        state = {
            'epoch': epoch, 'batch': batch, 'model_name': self.model.name, 'model_kwargs': self.model.init_kwargs,
            'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        ema_state = {'ema_state_dict': self.ema.module.state_dict()}
        name = f'model_{epoch}.pkl' if checkpoint else 'model.pkl'
        torch.save({**state, **ema_state}, self.run_dir / name)
        print_log(f'Model saved at {self.run_dir / name}')

    @torch.no_grad()
    def save_metric_plots(self):
        self.model.eval()
        df = self.train_metrics.read_log()
        if len(df) == 0:
            print_log('No metrics or plots to save')
            return None

        # Charts
        loss_names = list(filter(lambda col: 'loss' in col, df.columns))
        plot_lines(df, loss_names, title='Loss').savefig(self.run_dir / 'loss.pdf')
        df = self.val_metrics.read_log()
        alpha_names = list(filter(lambda col: 'alpha' in col, df.columns))
        colors = get_fancy_cmap()(np.linspace(0, 1, len(alpha_names) + 1)[1:])
        plot_lines(df, alpha_names, title='Opacity', colors=colors).savefig(self.run_dir / 'opacity.pdf')

        # Images / renderings
        rec = self.ema.module.predict(self.viz_samples, self.viz_labels, w_edges=True)
        self.rec_logger.save(rec)
        rec = self.ema.module.predict(self.viz_samples, self.viz_labels, filter_transparent=True)
        self.rec2_logger.save(rec)
        rec = self.ema.module.predict_synthetic(self.viz_samples, self.viz_labels)
        self.rec3_logger.save(rec)

        if not self.if_bin_render:
            self.txt_logger.save(self.model.get_arranged_block_txt())
            self.txt_logger.save_video(rmtree=LIGHT_MEMORY_RESULTS)
        print_log('Metrics and plots saved')

    def evaluate(self):
        self.model.eval()

        # qualitative
        out = path_mkdir(self.run_dir / 'quali_eval')
        self.ema.module.qualitative_eval(self.test_loader, self.device, path=out)

        # quantitative
        scores = self.ema.module.quantitative_eval(self.test_loader, self.device, hard_inference=True)
        print_log('final_scores: ' + ', '.join(["{}={:.5f}".format(k, v) for k, v in scores.items()]))
        with open(self.run_dir / 'final_scores.tsv', mode='w') as f:
            f.write("\t".join(scores.keys()) + "\n")
            f.write("\t".join(map('{:.5f}'.format, scores.values())) + "\n")

        # official DTU eval
        if self.dataset.name == 'dtu':
            scan_id = int(self.dataset.tag.replace('scan', ''))
            scale = self.dataset.scale_mat.to(self.device)

            # Blocks only
            scene = self.ema.module.build_blocks(filter_transparent=True, as_scene=True)
            verts, faces = scene.get_mesh_verts_faces(0)
            scene = Meshes((verts @ scale[:3, :3] + scale[:3, 3])[None], faces[None])
            evaluate_mesh(scene, scan_id, DATASETS_PATH / 'DTU', self.run_dir, save_viz=False)

        print_log('Evaluation over')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to train a NN model specified by a YML config')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-c', '--config', nargs='?', type=str, required=True, help='Config file name')
    parser.add_argument('-d', '--default', nargs='?', type=str, help='Default config file name')
    args = parser.parse_args()
    assert args.tag != '' and args.config != ''

    default_path = None if (args.default is None or args.default == '') else CONFIGS_PATH / args.default
    cfg = load_yaml(CONFIGS_PATH / args.config, default_path)
    seed, dataset = cfg['training'].get('seed', 4321), cfg['dataset']['name']
    if (RUNS_PATH / dataset / args.tag).exists():
        run_dir = RUNS_PATH  / args.config / args.tag
    else:
        run_dir = path_mkdir(RUNS_PATH / args.config / args.tag)
    run_dir = RUNS_PATH / dataset / args.tag
    create_logger(run_dir)
    dump_yaml(cfg, run_dir / Path(args.config).name)

    print_log(f'Trainer init: config_file={args.config}, run_dir={run_dir}')
    trainer = Trainer(cfg, run_dir)
    trainer.run(Exp_cfg=cfg['Exp_setting']['Exp_setting'])
