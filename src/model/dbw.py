from collections import OrderedDict
from copy import deepcopy
from toolz import valfilter
from pathlib import Path
from itertools import chain
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import itertools

import numpy as np
from pytorch3d.io import save_ply
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures.meshes import join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, random_rotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import get_loss, mse2psnr, tv_norm_funcs
from .renderer import Renderer, get_circle_traj, save_trajectory_as_video, save_mesh_as_video, DIRECTION_LIGHT
from .tools import safe_model_state_dict, elev_to_rotation_matrix, azim_to_rotation_matrix, roll_to_rotation_matrix
from utils import use_seed, path_mkdir
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.metrics import AverageMeter
from utils.mesh import get_icosphere, get_icosphere_uvs, save_mesh_as_obj, get_plane, point_to_uv_sphericalmap
from utils.plot import get_fancy_cmap
from utils.pytorch import torch_to, safe_pow, index
from utils.superquadric import parametric_sq, implicit_sq, sample_sq
from utils.logger import print_log

from model.SAM_Clustering import SAM_clustering, clustering


VIZ_SIZE = 256
DECIMATE_FACTOR = 8
OVERLAP_N_POINTS = 1000
OVERLAP_N_BLOCKS = 1.95
OVERLAP_TEMPERATURE = 0.005


class DifferentiableBlocksWorld(nn.Module):
    name = 'dbw'

    def __init__(self, img_size, **kwargs):
        super().__init__()
        self._init_kwargs = deepcopy(kwargs)
        self._init_kwargs['img_size'] = img_size
        self._init_blocks(**kwargs.get('mesh', {}), **kwargs.get('Exp_setting', {}))
        self._init_renderer(img_size, **kwargs.get('renderer', {}), **kwargs.get('Exp_setting', {}))
        self._init_rend_optim(**kwargs.get('rend_optim', {}))
        self._init_loss(**kwargs.get('loss', {}))
        self.cur_epoch = 0

    @property
    def init_kwargs(self):
        return deepcopy(self._init_kwargs)

    def _init_blocks(self, **kwargs):
        self.n_blocks = kwargs.pop('n_blocks', 1)
        self.vis_block_num = self.n_blocks
        self.S_world = kwargs.pop('S_world', 1)
        elev, azim, roll = kwargs.pop('R_world', [0, 0, 0])
        R_world = (elev_to_rotation_matrix(elev) @ azim_to_rotation_matrix(azim) @ roll_to_rotation_matrix(roll))[None]
        T_world = torch.Tensor(kwargs.pop('T_world', [0., 0., 0.]))[None]
        self.register_buffer('R_world', R_world)
        self.register_buffer('T_world', T_world)
        self.z_far = kwargs.pop('z_far', 10)
        self.ratio_block_scene = kwargs.pop('ratio_block_scene', 1 / 4)
        self.txt_size = kwargs.pop('txt_size', 256)
        self.txt_bkg_upscale = kwargs.pop('txt_bkg_upscale', 1)
        self.scale_min = kwargs.pop('scale_min', 0.2)
        opacity_init = kwargs.pop('opacity_init', 0.5)
        self.opacity_init = kwargs.pop('opacity_init', 0.5)
        T_range = kwargs.pop('T_range', [1, 1, 1])
        self.T_range = kwargs.pop('T_range', [1, 1, 1])
        T_init_mode = kwargs.pop('T_init_mode', 'gauss')

        self.if_bin_render = kwargs.pop('if_bin_render', True)

        #assert len(kwargs) == 0, kwargs

        self.bkg = get_icosphere(level=2, flip_faces=True).scale_verts_(self.z_far)
        self.register_buffer('bkg_verts_uvs', point_to_uv_sphericalmap(self.bkg.verts_packed()))
        self.ground = get_plane().scale_verts_(torch.Tensor([self.z_far, 1, self.z_far])[None])
        for k in range(3):
            self.ground = SubdivideMeshes()(self.ground)
        self.register_buffer('ground_verts_uvs', (self.ground.verts_packed()[:, [0, 2]] / self.z_far + 1) / 2)


        # Build primitive blocks
        block = get_icosphere(level=1)  # icosphere is different with superquadric, the unit icosphere is transferred to superquadric through the parametric equation
        self.blocks = join_meshes_as_batch([block.scale_verts(self.ratio_block_scene) for _ in range(self.n_blocks)])
        self.sq_eps = nn.Parameter(torch.zeros(self.n_blocks, 2))   # eps: epsilon1 & epsilon2
        verts = self.blocks.verts_padded() / self.ratio_block_scene
        self.register_buffer('sq_eta', torch.asin((verts[..., 1])))  # n(eta) in sq
        self.register_buffer('sq_omega', torch.atan2(verts[..., 0], verts[..., 2])) # w(omega) in sq
        # if not self.if_bin_render:
        faces_uvs, verts_uvs = get_icosphere_uvs(level=1, fix_continuity=True, fix_poles=True)
        p_left = abs(int(np.floor(verts_uvs.min(0)[0][0].item() * self.txt_size)))
        p_right = int(np.ceil((verts_uvs.max(0)[0][0].item() - 1) * self.txt_size))
        verts_u = (verts_uvs[..., 0] * self.txt_size + p_left) / (self.txt_size + p_left + p_right)
        verts_uvs = torch.stack([verts_u, verts_uvs[..., 1]], dim=-1)
        self.txt_padding = p_left, p_right
        self.BNF = len(faces_uvs)
        self.register_buffer('block_faces_uvs', faces_uvs)
        self.register_buffer('block_verts_uvs', verts_uvs)

        # Initialize learnable pose parameters
        self.R_6d_ground = nn.Parameter(torch.Tensor([[1., 0., 0., 0., 1., 0.]]))
        self.T_ground = nn.Parameter(torch.Tensor([[0., -0.9 * T_range[1], 0.]]))
        N = self.n_blocks
        S_init = (torch.rand(N, 3) + 0.5 - self.scale_min).log()  # scale corresponding to the x&y&z coor in the sq
        R_6d_init = matrix_to_rotation_6d(random_rotations(N))  # 6D rotation parametrization
        if T_init_mode == 'gauss':  # rotation matrix
            T_init = torch.randn(N, 3) / 2 * torch.Tensor(T_range)
        elif T_init_mode == 'uni':
            T_init = (2 * torch.rand(N, 3) - 1) * torch.Tensor(T_range)
        else:
            raise NotImplementedError
        self.S = nn.Parameter(S_init.clone())
        self.R_6d = nn.Parameter(R_6d_init.clone())
        self.T = nn.Parameter(T_init.clone())

        # Initialize learnable opacity and texture parameters
        self.alpha_logit = nn.Parameter(torch.logit(torch.ones(N) * opacity_init) + 1e-3)
        TS, txt_scale = self.txt_size, self.txt_bkg_upscale
        self.textures = torch.ones((N, TS, TS, 3))
        self.texture_bkg = torch.ones((1, TS * txt_scale, TS * txt_scale, 3))
        self.texture_ground = torch.ones((1, TS * txt_scale, TS * txt_scale, 3))

        if not self.if_bin_render:
            TS, txt_scale = self.txt_size, self.txt_bkg_upscale
            self.texture_bkg = nn.Parameter(torch.randn(1, TS * txt_scale, TS * txt_scale, 3) / 10)
            self.texture_ground = nn.Parameter(torch.randn(1, TS * txt_scale, TS * txt_scale, 3) / 10)
            self.textures = nn.Parameter(torch.randn(N, TS, TS, 3) / 10)

    def _init_rend_optim(self, **kwargs):
        # Basic
        self.opacity_noise = kwargs.pop('opacity_noise', False)
        self.decouple_rendering = kwargs.pop('decouple_rendering', False)
        self.coarse_learning = kwargs.pop('coarse_learning', True)
        self.decimate_txt = kwargs.pop('decimate_txt', False)
        self.decim_factor = kwargs.pop('decimate_factor', DECIMATE_FACTOR)
        self.kill_blocks = kwargs.pop('kill_blocks', False)

    def _init_renderer(self, img_size, **kwargs):
        self.renderer = Renderer(img_size, **kwargs)
        kwargs['sigma'] = 5e-6
        self.renderer_fine = Renderer(img_size, **kwargs)
        kwargs['faces_per_pixel'] = 1
        kwargs['sigma'] = 0
        kwargs['detach_bary'] = False
        self.renderer_env = Renderer(img_size, **kwargs)
        kwargs['lights'] = {'name': 'directional', 'direction': [DIRECTION_LIGHT], 'ambient_color': [[0.7, 0.7, 0.7]],
                            'diffuse_color': [[0.4, 0.4, 0.4]], 'specular_color': [[0., 0., 0.]]}
        kwargs['shading_type'] = 'flat'
        kwargs['background_color'] = (1, 1, 1)
        self.renderer_light = Renderer(img_size, **kwargs)

    def _init_loss(self, **kwargs):
        loss_weights = {
            'rgb': kwargs.pop('rgb_weight', 1.0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'parsimony': kwargs.pop('parsimony_weight', 0),
            'scale': kwargs.pop('scale_weight', 0),
            'tv': kwargs.pop('tv_weight', 0),
            'overlap': kwargs.pop('overlap_weight', 0),
        }
        blk_loss_weights = {
            'attn': kwargs.pop('attn_weight', 0),
        }
        name = kwargs.pop('name', 'mse')
        perceptual_name = kwargs.pop('perceptual_name', 'lpips')
        self.tv_norm = tv_norm_funcs[kwargs.pop('tv_type', 'l2sq')]
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.blk_loss_weights = valfilter(lambda v: v > 0, blk_loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)()
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss(perceptual_name)()

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch

    def step(self):
        self.cur_epoch += 1

    def to(self, device):
        super().to(device)
        # if not self.if_bin_render:
        self.bkg = self.bkg.to(device)
        self.ground = self.ground.to(device)
        self.blocks = self.blocks.to(device)
        self.renderer = self.renderer.to(device)
        self.renderer_fine = self.renderer_fine.to(device)
        self.renderer_env = self.renderer_env.to(device)
        self.renderer_light = self.renderer_light.to(device)
        self.textures = self.textures.to(device)
        self.texture_bkg = self.texture_bkg.to(device)
        self.texture_ground = self.texture_ground.to(device)
        return self

    @property
    def bkg_n_faces(self):
        return self.bkg.num_faces_per_mesh().sum().item()

    @property
    def ground_n_faces(self):
        return self.ground.num_faces_per_mesh().sum().item()

    @property
    def env_n_faces(self):
        return self.bkg_n_faces + self.ground_n_faces

    @property
    def blocks_n_faces(self):
        return self.blocks.num_faces_per_mesh().sum().item()

    @staticmethod
    def countNum(vert_list):
        unique_vert_elements = set()
        for sublist in vert_list:
            for tensor in sublist:
                unique_vert_elements.update(tensor.cpu().numpy())

        return len(unique_vert_elements)

    @staticmethod
    def newNumCount(vert_list, new_list):
        unique_vert_elements = set()
        for sublist in vert_list:
            for tensor in sublist:
                unique_vert_elements.update(tensor.cpu().numpy())

        new_elements = set()
        for tensor in new_list:
            new_elements.update(tensor.cpu().numpy())

        unique_new_elements = new_elements - unique_vert_elements
        return len(unique_new_elements)

    def get_scene_blk_rec(self, scene, data=None, bsz=1):
        bsz, gt, R_tgt, T_tgt = data['imgs'].shape[0], data['imgs'], data['R'], data['T']
        if 'K' in data and self.renderer.cameras.K is None:
            self.renderer.update_cameras(device=gt.device, K=data['K'][0:1])
            self.renderer_fine.update_cameras(device=gt.device, K=data['K'][0:1])
            self.renderer_env.update_cameras(device=gt.device, K=data['K'][0:1])
            self.renderer_light.update_cameras(device=gt.device, K=data['K'][0:1])
        fine_learning = not self.is_live('coarse_learning')
        renderer = self.renderer_fine if fine_learning else self.renderer

        rec = renderer(scene.extend(bsz), R=R_tgt, T=T_tgt).split([3, 1], dim=1)[0]
        return rec



    def get_vis_2dverts(self, data_loader=None, cover_rate=0.5, new_vert_per_round=3, viz=False, data=None, bsz=1, Debug_viz=False):
        blocks, blk_idx, scene = self.build_blocks(as_scene=True, sep_Blocks=True, synthetic_colors=True, filter_transparent=True)  #
        scene_verts, scene_rec, scene_mask,  _ = self.get_blk2dverts(data, scene.extend(bsz), scene=len(blk_idx))

        if viz:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colors = plt.cm.Spectral(np.linspace(0, 1, len(blocks)))
            color_index = 0
            for block in blocks:
                vertices_list = block.verts_list()
                col = colors[color_index]
                label = f'blk {color_index}'
                xyz = vertices_list[0].cpu()
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=col, label=label)
                color_index += 1
            ax.set_xlim([0.5, -0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([-0.5, 0.5])
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.legend()
            ax.grid(True)

            plt.show()

        Vis_verts = {}
        Verts_idx = {}
        Rec = {}
        Data = {}
        for block_idx, block in zip(blk_idx, blocks):
            if data_loader is not None:
                vis_verts = []
                verts_idx = []
                mask = []
                data = []
                for data_idx, (_data, _) in enumerate(data_loader):
                    if self.countNum(verts_idx) >= (42 * cover_rate):
                        break
                    _verts2D, _mask, _vert_idx = self.get_blk2dverts(torch_to(_data, block.device), block)
                    if self.newNumCount(verts_idx, _vert_idx) < new_vert_per_round:
                        continue
                    vis_verts.append(_verts2D)
                    verts_idx.append(_vert_idx)
                    mask.append(_mask)
                    data.append(_data['imgs'])
            else:
                assert len(scene_verts) == bsz
                vis_verts, rec, _, verts_idx = self.get_blk2dverts(data, block.extend(bsz))
                matching_mask = torch.isclose(scene_rec, rec, atol=1e-6, rtol=1e-4).all(dim=1)  # Reduce across the channel dimension
                for batch_idx, (verts, v_idx) in enumerate(zip(vis_verts, verts_idx)):
                    mask = matching_mask[batch_idx]
                    try:
                        verts_matches = mask[verts.to(int).long()[:, 1], verts.to(int).long()[:, 0]]
                        vis_verts[batch_idx] = verts[verts_matches]
                        verts_idx[batch_idx] = v_idx[verts_matches]
                    except:
                        vis_verts[batch_idx] = [None]
                        verts_idx[batch_idx] = [None]
                        continue
            if block_idx is None:
                Vis_verts = vis_verts
                Verts_idx = verts_idx
                Rec = rec
                Data = data
            else:
                Vis_verts[int(block_idx)] = vis_verts
                Verts_idx[int(block_idx)] = verts_idx
                Rec[int(block_idx)] = rec
                Data[int(block_idx)] = data

        if Debug_viz:
            resulting_list = []
            for i in range(bsz):
                concatenated_lists = [item for key in Vis_verts for item in Vis_verts[key][i]]
                tensor = torch.cat([item.unsqueeze(0) for item in concatenated_lists], dim=0)
                resulting_list.append(tensor)
            blocks, blk_idx, scene = self.build_blocks(as_scene=True, sep_Blocks=True)  #
            scene_verts, scene_rec, scene_mask, _ = self.get_blk2dverts(data, scene.extend(bsz), scene=len(blk_idx), viz=Debug_viz, draw_point=resulting_list)
            return scene_rec, resulting_list
        return Vis_verts, Verts_idx, Rec, blocks, scene_rec

    def get_blk2dverts(self, data, block, scene=1, viz=False, draw_point=None):
        bsz, gt, R_tgt, T_tgt = data['imgs'].shape[0], data['imgs'], data['R'], data['T']
        if 'K' in data and self.renderer.cameras.K is None:
            self.renderer.update_cameras(device=gt.device, K=data['K'][0:1])
            self.renderer_fine.update_cameras(device=gt.device, K=data['K'][0:1])
            self.renderer_env.update_cameras(device=gt.device, K=data['K'][0:1])
            self.renderer_light.update_cameras(device=gt.device, K=data['K'][0:1])
        fine_learning = not self.is_live('coarse_learning')
        renderer = self.renderer_fine if fine_learning else self.renderer

        rec, mask = renderer(block, R=R_tgt, T=T_tgt).split([3, 1], dim=1)
        if viz:
            colors = self.get_scene_face_colors(filter_transparent=True).repeat(bsz, 1)
            rec = renderer.draw_edges(rec, block, R_tgt, T_tgt, colors=colors)
            rec, vis_verts = renderer.draw_verts(rec, block[0], R_tgt, T_tgt, colors=colors, bsz=bsz, scene=scene, draw_verts=draw_point)
            return vis_verts, rec, None, None
        else:
            vis_verts, vert_idx = renderer.draw_verts(rec, block[0], R_tgt, T_tgt, bsz=bsz, dry_draw=True, scene=scene)
        return vis_verts, rec, mask, vert_idx

    def forward(self, inp, labels):
        rec = self.predict(inp, labels)
        return self.compute_losses(inp['imgs'], rec)

    def predict(self, inp, labels, w_edges=False, filter_transparent=False, w_vert=False, **Exp_cfg):
        sep_Blocks = Exp_cfg.get('separate_blocks', False)
        B, gt, R_tgt, T_tgt = len(inp['imgs']), inp['imgs'], inp['R'], inp['T']
        if 'K' in inp and self.renderer.cameras.K is None:
            self.renderer.update_cameras(device=gt.device, K=inp['K'][0:1])
            self.renderer_fine.update_cameras(device=gt.device, K=inp['K'][0:1])
            self.renderer_env.update_cameras(device=gt.device, K=inp['K'][0:1])
            self.renderer_light.update_cameras(device=gt.device, K=inp['K'][0:1])

        fine_learning = not self.is_live('coarse_learning')
        filter_tsp = filter_transparent or fine_learning
        renderer = self.renderer_fine if fine_learning else self.renderer

        if sep_Blocks and self.if_bin_render:
            blocks, _, _ = self.build_blocks(filter_transparent=filter_tsp, as_scene=True, sep_Blocks=True)   # world_coord
            rec = torch.empty((B, len(blocks), 3, renderer.img_size[0], renderer.img_size[1],)).to(self.blocks.device)
            mask = torch.empty((B, len(blocks), 1, renderer.img_size[0], renderer.img_size[1],)).to(self.blocks.device)
            vis_verts = []
            for idx, block in enumerate(blocks):
                _rec, _mask = renderer(block.extend(B), R=R_tgt, T=T_tgt).split([3, 1], dim=1)
                if w_edges:
                    _rec = renderer.draw_edges(_rec, block.extend(B), R_tgt, T_tgt, colors=colors)
                if w_vert:
                    _rec, _vis_verts = renderer.draw_verts(_rec, block, R_tgt, T_tgt, colors=colors, bsz=B)
                rec[:, idx, :, :, :] = _rec
                mask[:, idx, :, :, :] = _mask
                vis_verts.append(_vis_verts)
            transposed_list = [list(row) for row in zip(*vis_verts)]
            return rec, transposed_list

        elif self.decouple_rendering:
            if not self.if_bin_render:
                env = join_meshes_as_scene([self.build_bkg(world_coord=True), self.build_ground(world_coord=True)])
                rec_env = self.renderer_env(env.extend(B), R=R_tgt, T=T_tgt).split([3, 1], dim=1)[0]
            blocks = self.build_blocks(filter_transparent=filter_tsp, as_scene=True)
            if len(blocks) > 0:
                alpha = None if filter_tsp else self._alpha.repeat_interleave(self.BNF).repeat(B)
                rec_fg, mask = renderer(blocks.extend(B), R=R_tgt, T=T_tgt, faces_alpha=alpha).split([3, 1], dim=1)
            else:
                rec_fg, mask = torch.zeros_like(rec_env), torch.zeros_like(rec_env)
            rec = rec_fg * mask + (1 - mask) * rec_env

        else:
            scene = self.build_scene(filter_transparent=filter_tsp)
            if not filter_tsp:
                if not self.if_bin_render:
                    alpha_env = torch.ones(self.env_n_faces, device=gt.device)
                    alpha = torch.cat([alpha_env, self._alpha.repeat_interleave(self.BNF)], dim=0).repeat(B)
                else:
                    alpha = torch.cat([self._alpha.repeat_interleave(self.BNF)], dim=0).repeat(B)
            else:
                alpha = None
            rec, mask = renderer(scene.extend(B), R=R_tgt, T=T_tgt, faces_alpha=alpha).split([3, 1], dim=1)

        colors = self.get_scene_face_colors(filter_transparent=filter_tsp).repeat(B, 1)
        if w_edges:
            if self.decouple_rendering:
                scene = join_meshes_as_scene([env, blocks]) if len(blocks) > 0 else env
            rec = renderer.draw_edges(rec, scene.extend(B), R_tgt, T_tgt, colors=colors)
        return rec

    def predict_synthetic(self, inp, labels):
        B, R_tgt, T_tgt = len(inp['imgs']), inp['R'], inp['T']
        blocks = self.build_blocks(filter_transparent=True, synthetic_colors=True, as_scene=True)
        if len(blocks) > 0:
            rec = self.renderer_light(blocks.extend(B), R=R_tgt, T=T_tgt, viz_purpose=True)[:, :3]
        else:
            rec = torch.ones_like(inp['imgs'])
        return rec

    def build_scene(self, filter_transparent=False, w_bkg=True, w_ground=True, reduce_ground=False, synthetic_colors=False, save=False):
        meshes = []
        if self.if_bin_render and not save:
            w_bkg = False
            w_ground = False
        if w_bkg:
            meshes.append(self.build_bkg(synthetic_colors=synthetic_colors))
        if w_ground:
            meshes.append(self.build_ground(reduced=reduce_ground, synthetic_colors=synthetic_colors))
        blocks = self.build_blocks(filter_transparent, synthetic_colors=synthetic_colors)[0]
        if len(blocks) > 0:
            meshes.append(blocks)
        N_meshes = len(meshes) - 1 + len(blocks)
        if N_meshes > 1:
            scene = join_meshes_as_scene(meshes)
        else:
            scene = meshes[0] if len(meshes) > 0 else self.build_bkg(synthetic_colors=synthetic_colors)
        verts, faces = scene.get_mesh_verts_faces(0)
        verts = (verts[None] * self.S_world) @ self.R_world + self.T_world[:, None]
        return Meshes(verts, faces[None], scene.textures)

    def build_bkg(self, reduced=False, world_coord=False, synthetic_colors=False):
        verts, faces = [t[None] for t in self.bkg.get_mesh_verts_faces(0)]
        if reduced:
            verts = verts * 3 / self.z_far
        if world_coord:
            verts = (verts * self.S_world) @ self.R_world + self.T_world[:, None]
        maps = torch.sigmoid(self.texture_bkg) if not synthetic_colors else torch.ones_like(self.texture_bkg)
        self._bkg_maps = maps
        # Regularization
        if self.training and self.is_live('decimate_txt'):
            sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), kernel_size=self.decim_factor, stride=self.decim_factor)
            maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        return Meshes(verts, faces, textures=TexturesUV(maps, faces, self.bkg_verts_uvs[None], align_corners=True))

    def build_ground(self, reduced=False, world_coord=False, synthetic_colors=False):
        S_ground = 1. if not reduced else torch.Tensor([3 / self.z_far, 1, 3 / self.z_far]).to(self.bkg.device)
        verts, faces = [t[None] for t in self.ground.get_mesh_verts_faces(0)]
        verts = (verts * S_ground) @ rotation_6d_to_matrix(self.R_6d_ground) + self.T_ground[:, None]
        if world_coord:
            verts = (verts * self.S_world) @ self.R_world + self.T_world[:, None]
        maps = torch.sigmoid(self.texture_ground) if not synthetic_colors else torch.ones_like(self.texture_ground)
        self._ground_maps = maps
        # Regularization
        if self.training and self.is_live('decimate_txt'):
            sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), kernel_size=self.decim_factor, stride=self.decim_factor)
            maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        return Meshes(verts, faces, textures=TexturesUV(maps, faces, self.ground_verts_uvs[None], align_corners=True))

    def build_blocks(self, filter_transparent=False, world_coord=False, as_scene=False, synthetic_colors=False, sep_Blocks=False):
        coarse_learning = self.training and self.is_live('coarse_learning')
        S, R, T = self.S.exp() + self.scale_min, rotation_6d_to_matrix(self.R_6d), self.T
        if self.opacity_noise and coarse_learning:
            alpha_logit = self.alpha_logit + self.opacity_noise * torch.randn_like(self.alpha_logit)
        else:
            alpha_logit = self.alpha_logit
        self._alpha = torch.sigmoid(alpha_logit)
        self._alpha_full = self._alpha.clone()
        maps = self.textures if self.if_bin_render else torch.sigmoid(self.textures)
        verts = (self.get_blocks_verts() * S[:, None]) @ R + T[:, None]
        faces = self.blocks.faces_padded()

        vis_blk_idx = torch.arange(self.n_blocks).to(self.blocks.device)
        # Filter blocks based on opacities
        if filter_transparent or self.kill_blocks:
            if filter_transparent:
                mask = torch.sigmoid(self.alpha_logit) > 0.5
            else:
                mask = torch.sigmoid(self.alpha_logit) > 0.2
            self._alpha_full = self._alpha_full * mask
            NB = sum(mask).item()
            vis_blk_idx = torch.where(mask)[0]
            if NB == 0:
                verts, faces, maps = [], [], []
            else:
                verts, faces, maps, self._alpha = verts[mask], faces[mask], maps[mask], self._alpha[mask]
        else:
            NB = self.n_blocks
        self.vis_block_num = NB
        if synthetic_colors:
            values = torch.linspace(0, 1, NB + 1)[1:]
            colors = torch.from_numpy(get_fancy_cmap()(values.cpu().numpy())).float()
            maps = colors[:, None, None].expand(-1, self.txt_size, self.txt_size, -1).to(maps.device)
        self._blocks_maps, self._blocks_SRT = maps, (S, R, T)
        # Regularization
        if len(maps) > 0 and coarse_learning and not self.if_bin_render:
            if self.is_live('decimate_txt'):
                sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), self.decim_factor, stride=self.decim_factor)
                maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        # Build textures and meshes object
        verts_uvs = self.block_verts_uvs[None].expand(self.n_blocks, -1, -1)[:NB] if NB != 0 else []
        faces_uvs = self.block_faces_uvs[None].expand(self.n_blocks, -1, -1)[:NB] if NB != 0 else []
        if len(maps) > 0:
            p_left, p_right = self.txt_padding
            maps = F.pad(maps.permute(0, 3, 1, 2), pad=(p_left, p_right, 0, 0), mode='circular').permute(0, 2, 3, 1)
        txt = TexturesUV(maps, faces_uvs, verts_uvs, align_corners=True)
        if (world_coord or as_scene) and len(verts) > 0:
            verts = (verts * self.S_world) @ self.R_world + self.T_world[:, None]
        blocks = Meshes(verts, faces, textures=txt)
        if as_scene and len(blocks) > 0 and not sep_Blocks:
            try:
                return join_meshes_as_scene(blocks)
            except ValueError:
                return blocks
        else:
            try:
                return [blocks, vis_blk_idx, join_meshes_as_scene(blocks)]
            except ValueError:
                return [blocks, vis_blk_idx, blocks]

    def get_blocks_verts(self):
        eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
        verts = parametric_sq(self.sq_eta, self.sq_omega, eps1, eps2) * self.ratio_block_scene
        self._blocks_eps = eps1, eps2
        return verts

    def sample_points_from_blocks(self, N_points=500):
        eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
        S, R, T = self.S.exp() + self.scale_min, rotation_6d_to_matrix(self.R_6d), self.T
        points = sample_sq(eps1, eps2, scale=S * self.ratio_block_scene, N_points=N_points)  # NP3
        points = points @ R + T[:, None]
        return points

    def compute_losses(self, imgs, rec):
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}

        coarse_learning = self.is_live('coarse_learning')
        # Pixel-wise reconstrution error on RGB values
        if 'rgb' in losses:
            losses['rgb'] = self.loss_weights['rgb'] * self.criterion(imgs, rec)  # use MSE here
        # Perceptual loss
        if 'perceptual' in losses:
            factor = 1 if coarse_learning else 0.1
            losses['perceptual'] = self.loss_weights['perceptual'] * factor * self.perceptual_loss(imgs, rec)
        # Parsimony
        if 'parsimony' in losses:
            factor = 1 if coarse_learning else 0
            alpha = self._alpha_full  # if coarse_learning else self._alpha_full[self._alpha_full > 0.5]#(self._alpha_full > 0.5).float()
            losses['parsimony'] = self.loss_weights['parsimony'] * safe_pow(alpha, 0.5).mean() * factor
        # TV loss
        if 'tv' in losses:
            factor = 1 if coarse_learning else 0.1
            tv_loss = sum([self.tv_norm(torch.diff(self._bkg_maps, dim=k)).mean() for k in [1, 2]])
            if len(self._blocks_maps) > 0:
                # we use mapping continuity in TV
                dx = self.tv_norm(torch.diff(self._blocks_maps, dim=2, append=self._blocks_maps[:, :, 0:1]))
                dy = self.tv_norm(torch.diff(self._blocks_maps, dim=1))
                tv_loss += (dx.sum(0).mean() + dy.sum(0).mean())  # sum over blocks so that each map receives same grad
            tv_loss += sum([self.tv_norm(torch.diff(self._ground_maps, dim=k)).mean() for k in [1, 2]]) * factor
            losses['tv'] = self.loss_weights['tv'] * factor * tv_loss
        # Overlap
        if 'overlap' in losses:
            # factor = 1 if coarse_learning else 0
            N = self.n_blocks
            with torch.no_grad():
                points = torch.rand(N, OVERLAP_N_POINTS, 3, device=rec.device) * 2 - 1
                S, R, T = self._blocks_SRT
                points = (points * self.ratio_block_scene * S[:, None]) @ R + T[:, None]
                points = points.view(-1, 3)[None].expand(N, -1, -1)

            eps1, eps2 = self._blocks_eps
            points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene) # ratio_block_scene=1/4
            sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
            occupancy = torch.sigmoid(-sdf / OVERLAP_TEMPERATURE)
            alpha = self._alpha_full if coarse_learning else (self._alpha_full > 0.5).float()
            occupancy = occupancy * alpha[:, None]
            overlap_loss = (occupancy.sum(0) - OVERLAP_N_BLOCKS).clamp(0).mean()
            losses['overlap'] = self.loss_weights['overlap'] * overlap_loss

        losses['total'] = sum(losses.values())
        return losses

    def compute_blk_losses(self, imgs, Exp_cfg, split_blk=False, combine_blk=False):
        blk_losses = {}
        blk_idx_split = {}
        blkidx2idx_map = {} # map the parameter_idx to idx inside the blocks
        idx2blkidx_map = {}

        attn_type = {
            'attn': 0,
            'attn_fin': 1,
            'attn_sum': 2
        }
        attn_type = attn_type[Exp_cfg.pop('attn_type', 'attn_sum')]
        blk_loss_type = Exp_cfg.pop('blk_loss_type', 'L2')
        dis_ratio_thres = Exp_cfg.pop('dis_ratio_thres', 0.6)
        num_ratio_thres = Exp_cfg.pop('num_ratio_thres', 0.7)
        blk_loss_thres = Exp_cfg.pop('blk_loss_thres', [0.01, 0.04])
        blk_loss_thres[0], blk_loss_thres[1] = blk_loss_thres[0] * self.blk_loss_weights['attn'], blk_loss_thres[1] * self.blk_loss_weights['attn']
        bsz = imgs['imgs'].shape[0]
        device = imgs['imgs'].device
        img_shape = imgs['imgs'].size()[2:]

        vis_verts, verts_idx, mask, blocks, scene = self.get_vis_2dverts(data=imgs, bsz=bsz)
        blks_cluster, blk_cluster_labels, blk_center_coord, blk_bbx, blk_center_verts_idx, blk_attn = SAM_clustering(vis_verts, verts_idx, mask, imgs['imgs'], bsz=bsz, attn_type= attn_type)
        for idx, ((blk_idx, labels), (_, coord), (_, verts), (_, bbx), (_, center_verts_idx), (_, blk_rec)) in enumerate(zip(blk_cluster_labels.items(), blk_center_coord.items(), vis_verts.items(), blk_bbx.items(), blk_center_verts_idx.items(), mask.items())):
            blkidx2idx_map[blk_idx] = idx
            idx2blkidx_map[idx] = blk_idx
            blk_losses[blk_idx] = self.blk_loss(labels, coord, center_verts_idx, verts, bsz, blk_loss_type, img_shape, blocks[idx])
            is_split, batch_split, split_label = self.is_split(bbx, coord, bsz, dis_ratio_thres, num_ratio_thres, device)
            if (is_split and blk_losses[blk_idx].item() >= blk_loss_thres[0]) or blk_losses[blk_idx].item() >= blk_loss_thres[1]:
                blk_idx_split[blk_idx] = [verts_idx[blk_idx][batch_split], labels[batch_split], split_label, center_verts_idx[batch_split]]

        if split_blk or combine_blk:
            if combine_blk:
                combine_pair = self.is_combine(imgs, blocks, blkidx2idx_map, idx2blkidx_map, bsz, mask, scene, blk_attn, vis_verts)
                if combine_pair is not None:
                    scene = (scene != 0.).float()
                    _scene = self.pop_blocks(blocks, [blkidx2idx_map[combine_pair[0]], blkidx2idx_map[combine_pair[1]]])
                    _scene_mesh = join_meshes_as_scene(_scene)
                    _scene_rec = self.get_scene_blk_rec(data=imgs, bsz=bsz, scene=_scene_mesh)
                    is_disabled = True if (scene == _scene_rec).sum() / (bsz * 3 * img_shape[0] * img_shape[1]) >= 0.95 else False
                    self.combine_blk(combine_pair, disabled=is_disabled)
                return blk_losses, combine_pair

            if split_blk and len(blk_idx_split) > 0: # split the blocks
                for split_blk_idx, split_vert_idx in blk_idx_split.items():
                    self.split_blk(blocks[blkidx2idx_map[split_blk_idx]], split_blk_idx, split_vert_idx, bsz)

        return blk_losses

    def pop_blocks(self, meshes, indices_to_remove):
        # Validate indices
        indices_to_remove = set(indices_to_remove)  # Remove duplicates and allow for efficient checking
        if any(index < 0 or index >= len(meshes) for index in indices_to_remove):
            raise IndexError("One or more indices are out of range")

        # Extract mesh components
        verts_list = meshes.verts_list()
        faces_list = meshes.faces_list()

        # Remove the meshes at the specified indices
        new_verts_list = [v for i, v in enumerate(verts_list) if i not in indices_to_remove]
        new_faces_list = [f for i, f in enumerate(faces_list) if i not in indices_to_remove]

        textures = TexturesVertex(torch.full([len(new_verts_list), 42, 3], fill_value=1.)).to(meshes.device)
        new_meshes = Meshes(verts=new_verts_list, faces=new_faces_list, textures=textures)
        return new_meshes

    def combine_blk(self, pair_idx, disabled=False):
        print_log(f'***combing block pair {pair_idx}***')
        with torch.no_grad():
            self.alpha_logit[pair_idx[0]], self.alpha_logit[pair_idx[1]] = -1e1, -1e1
        if disabled:
            return
        new_alpha_logit = torch.logit(torch.ones(1) * 0.5) + 1e-3
        new_sq_eps = ((self.sq_eps[pair_idx[0]] + self.sq_eps[pair_idx[1]]) / 2).unsqueeze(0)
        new_S = ((self.S[pair_idx[0]] + self.S[pair_idx[1]]) / 2).unsqueeze(0) # scale corresponding to the x&y&z coor in the sq
        new_R_6d = ((self.R_6d[pair_idx[0]] + self.R_6d[pair_idx[1]]) / 2).unsqueeze(0) # 6D rotation parametrization
        new_T = ((self.T[pair_idx[0]] + self.T[pair_idx[1]]) / 2).unsqueeze(0)

        new_sq_eps = self.init_new_param(self.sq_eps, [new_sq_eps.to(self.sq_eps.device)])
        new_S = self.init_new_param(self.S, [new_S.to(self.sq_eps.device)])
        new_R_6d = self.init_new_param(self.R_6d, [new_R_6d.to(self.sq_eps.device)])
        new_T = self.init_new_param(self.T, [new_T.to(self.sq_eps.device)])
        new_alpha_logit = self.init_new_param(self.alpha_logit, [new_alpha_logit.to(self.sq_eps.device)])

        self.n_blocks += 1
        self.sq_eps = torch.nn.Parameter(new_sq_eps)
        self.S = torch.nn.Parameter(new_S)
        self.R_6d = torch.nn.Parameter(new_R_6d)
        self.T = torch.nn.Parameter(new_T)
        self.alpha_logit = torch.nn.Parameter(new_alpha_logit)


        self.blocks = join_meshes_as_batch([self.blocks, self.blocks[pair_idx[0]]])
        self.sq_eta = torch.cat((self.sq_eta, self.sq_eta[pair_idx[0], None]), dim=0)
        self.sq_omega = torch.cat((self.sq_omega, self.sq_omega[pair_idx[0], None]), dim=0)
        self.textures = torch.ones((self.n_blocks, self.txt_size, self.txt_size, 3)).to(self.blocks.device)

        return

    def is_combine(self, data, blocks, blk2idx_map, idx2blk_map, bsz, rec, scene_rec, attn, vis_verts):    # blocks is colored
        for idx, blk_idx in idx2blk_map.items():
            neighbor_blk_idx = self.find_neighbor(idx, blk_idx, blocks, idx2blk_map)
            for i in range(bsz):
                matrix = attn[blk_idx][i] + attn[neighbor_blk_idx][i]
                cluster = clustering(matrix)
                if cluster is None:
                    continue
                if len(cluster.exemplars_) != 0:
                    break
                return [blk_idx, neighbor_blk_idx]

        return None

    def find_neighbor(self, idx, blk_idx, blocks, idx2blk_map):
        min_distance = float('inf')
        block = blocks[idx]
        neighbor_blk_idx = None

        for idx, b in enumerate(blocks):
            if blk_idx == idx2blk_map[idx]:
                continue
            distance = self.mesh_distance(block, b)
            if distance < min_distance:
                min_distance = distance
                neighbor_blk_idx = idx2blk_map[idx]

        return neighbor_blk_idx

    @staticmethod
    def mesh_distance(mesh1, mesh2):
        # Calculate the mean Euclidean distance between vertices of the two meshes
        distance = torch.norm(mesh1.verts_packed() - mesh2.verts_packed(), dim=1).mean()
        return distance.item()

    def split_blk(self, block, split_blk_idx, split_vert_idx, bsz, scale_rate=0.4, T_rate=0.99, alpha_rate=0.6, eps_rate=0.8): # split the individual block, split_blk_idx align with the parameters
        world_verts = (block.extend(bsz)).verts_packed()  # world_verts = [verts * S] @ R + T
        org_verts_mean = (torch.mean(self.get_blocks_verts()[split_blk_idx], dim=0)).unsqueeze(0)
        S, R, T = self.S.exp()[split_blk_idx] + self.scale_min, rotation_6d_to_matrix(self.R_6d)[split_blk_idx], self.T[split_blk_idx]
        tgt_S = S * scale_rate

        split_idx, full_label, main_label, center_idx = split_vert_idx
        assert len(main_label) == 2

        tgt0_world_mean, tgt1_world_mean = world_verts[center_idx[0]], world_verts[center_idx[1]]

        # move the org_verts_mean to the target_mean position
        _verts = (tgt0_world_mean - self.T_world) @ torch.inverse(self.R_world) / self.S_world
        T0 = [_verts[0] - (org_verts_mean * tgt_S) @ R][0] * T_rate + T * (1-T_rate)
        _verts = (tgt1_world_mean - self.T_world) @ torch.inverse(self.R_world) / self.S_world
        T1 = [_verts[0] - (org_verts_mean * tgt_S) @ R][0] * T_rate + T * (1-T_rate)
        # print_log(f'*** splitting the block{split_blk_idx} ***')
        new_sq_eps = self.sq_eps[split_blk_idx] * eps_rate
        new_sq_eps = self.init_new_param(self.sq_eps, [new_sq_eps, new_sq_eps])
        new_S = torch.log(tgt_S - self.scale_min)
        new_S = torch.where(torch.isnan(new_S), torch.tensor(-2.).to(S.device), new_S)
        new_S = self.init_new_param(self.S, [new_S, new_S])
        new_R_6d = self.init_new_param(self.R_6d, [self.R_6d[split_blk_idx], self.R_6d[split_blk_idx]])
        new_T = self.init_new_param(self.T, [T0[0], T1[0]])
        tgt_alpha = torch.sigmoid(self.alpha_logit)[split_blk_idx] * alpha_rate
        new_alpha_logit = torch.log(tgt_alpha / (1 - tgt_alpha))
        new_alpha_logit = self.init_new_param(self.alpha_logit, [new_alpha_logit, new_alpha_logit])

        self.n_blocks += 2
        self.sq_eps = torch.nn.Parameter(new_sq_eps)
        self.S = torch.nn.Parameter(new_S)
        self.R_6d = torch.nn.Parameter(new_R_6d)
        self.T = torch.nn.Parameter(new_T)
        self.alpha_logit = torch.nn.Parameter(new_alpha_logit)
        with torch.no_grad():
            self.alpha_logit[split_blk_idx] = -1e1

        self.blocks = join_meshes_as_batch([self.blocks, self.blocks[split_blk_idx], self.blocks[split_blk_idx]])
        self.sq_eta = torch.cat((self.sq_eta, self.sq_eta[split_blk_idx, None], self.sq_eta[split_blk_idx, None]), dim=0)
        self.sq_omega = torch.cat((self.sq_omega, self.sq_omega[split_blk_idx, None], self.sq_omega[split_blk_idx, None]), dim=0)
        self.textures = torch.ones((self.n_blocks, self.txt_size, self.txt_size, 3)).to(self.blocks.device)
        return

    def init_new_param(self, param, new_value_list):
        assert param.dim() == 1 or param.dim() == 2
        if len(new_value_list) == 2:
            value1, value2 = new_value_list
            new_param = torch.cat((param, value1[None], value2[None]), dim=0)
        else:
            assert len(new_value_list) == 1
            value1 = new_value_list[0]
            new_param = torch.cat((param, value1), dim=0)
        return new_param

    def blk_loss(self, labels, coord, vert_idx, verts, bsz, blk_loss_type, img_shape, block):
        device = self.blocks.device
        labels_num = []
        main_cluster_label = []
        for arr in (sublist for sublist in labels):
            if arr is None or np.all(arr == -1):
                labels_num.append(0)
                main_cluster_label.append(-1)
            else:
                unique_numbers, counts = np.unique(arr[arr != -1], return_counts=True)
                labels_num.append(len(unique_numbers))
                main_cluster_label.append(unique_numbers[np.argmax(counts)])
        # [0 if np.all(arr == -1) else len(np.unique(arr[arr != -1])) for arr in (sublist[attn_type] for sublist in labels)]
        if np.all(np.array(labels_num) == 0):
            return torch.zeros(1).cuda().requires_grad_()

        img_shape = torch.tensor([img_shape[-1], img_shape[0]]).to(device=device, dtype=torch.float32)
        coord = [sublist for sublist in coord]
        loss = torch.empty((bsz, 1)).to(device)
        None_num = 0.
        for idx, label in enumerate(main_cluster_label):
            if label == -1:
                loss[idx] = torch.zeros(1).to(device)
                continue
            elif label is None:
                loss[idx] = torch.zeros(1).to(device)
                None_num += 1
                continue
            else:
                center_coord = coord[idx][label]
                mask = torch.tensor(labels[idx]) != label
                vert = verts[idx][mask]
                if blk_loss_type == 'L2':
                    l = F.mse_loss(vert / img_shape, center_coord.expand_as(vert) / img_shape, reduction='mean')
                    loss[idx] = l
        # loss = self.loss_weights['attn'] * safe_pow(torch.tensor(labels_num).to(device), 0.5).mean()
        if None_num == 0:
            loss = self.blk_loss_weights['attn'] * loss.mean()
        else:
            loss = self.blk_loss_weights['attn'] * (loss.sum()/(bsz - None_num))
        return loss

    def get_center_distance(self, coord):
        distances = []
        None_count = 0.
        max_pair_indices = []
        # Iterate through coord and calculate distances or append -1
        for item in coord:
            if item == -1 or item is None:
                # Append -1 directly without calculation
                distances.append(-1)
                max_pair_indices.append(-1)
                None_count += 1.
            else:
                if len(item) == 2:
                    # Calculate the Euclidean distance between the two tensors
                    distance = torch.norm(item[0] - item[1]).item()  # .item() to convert to Python scalar
                    distances.append(distance)
                    max_pair_indices.append((0, 1))
                else:
                    max_dist = 0
                    pair_indices = (0, 0)
                    for item1, item2 in itertools.combinations(item, 2):
                        dist = torch.norm(item1 - item2).item()
                        if dist > max_dist:
                            max_dist = dist
                        # itertools.combinations returns pairs in order, so indices are straightforward
                            pair_indices = (index(item, item1), index(item, item2))
                    max_pair_indices.append(pair_indices)
                    distances.append(max_dist)


        return distances, max_pair_indices, None_count

    def is_split(self, bbx, coord, bsz, dis_ratio_thres, num_ratio_thres, device):
        thres = [dis_ratio_thres * math.sqrt((sublist[2] - sublist[0]) ** 2 + (sublist[3] - sublist[1]) ** 2) if sublist is not None else 0. for sublist in bbx] # bbx:[X_min, Y_min, X_max, Y_max]
        distance, pair_indices, None_count = self.get_center_distance(coord)
        batch_split = np.argmax(np.array(distance))
        if None_count <= bsz * (1. - num_ratio_thres) and (np.array(distance) > np.array(thres)).sum() >= (bsz - None_count):
            return True, batch_split, pair_indices[batch_split]
        else:
            return False, batch_split, pair_indices[batch_split]

    def get_opacities(self):
        alpha = torch.sigmoid(self.alpha_logit)
        if self.kill_blocks:
            alpha = alpha * (alpha > 0.01)
        return alpha

    @torch.no_grad()
    def get_nb_opaque_blocks(self):
        return (self.get_opacities() > 0.5).sum().item()

    @torch.no_grad()
    def get_scene_face_colors(self, filter_transparent=False, w_env=True):
        if self.if_bin_render:
            w_env = False
        val_blocks = torch.linspace(0, 1, self.vis_block_num + 1)[1:]

        cmap = get_fancy_cmap()
        # NFE = self.env_n_faces if w_env else 0
        NFE = 0 if not w_env else self.env_n_faces
        values = torch.cat([torch.zeros(NFE), val_blocks.repeat_interleave(self.BNF)])
        colors = cmap(values.numpy())
        return torch.from_numpy(colors).float().to(self.blocks.device)

    @torch.no_grad()
    def get_arranged_block_txt(self):
        maps = torch.sigmoid(self.textures).permute(0, 3, 1, 2)
        ncol, nrow = 5, len(maps) // 5
        rows = [torch.cat([maps[k] for k in range(ncol * i, ncol * (i + 1))], dim=2) for i in range(nrow)]
        return torch.cat(rows, dim=1)[None]

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            name = name.replace('spq_', 'sq_')  # Backward compatibility
            if name in state:
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    def is_live(self, name):
        milestone = getattr(self, name)
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    @torch.no_grad()
    def quantitative_eval(self, loader, device, hard_inference=True):
        self.eval()
        opacities = self.get_opacities()
        n_blocks = (opacities > 0.5).sum().item()

        mse_func = get_loss('mse')().to(device)
        ssim_func = get_loss('ssim')(padding=False).to(device)
        lpips_func = get_loss('lpips')().to(device)
        loss_tot, loss_rec, psnr, ssim, lpips = [AverageMeter() for _ in range(5)]
        scene = self.build_scene(filter_transparent=True)
        for j, (inp, labels) in enumerate(loader):
            inp = torch_to(inp, device)
            imgs, N = inp['imgs'], len(inp['imgs'])
            if hard_inference:
                rec = self.renderer(scene.extend(N), inp['R'], inp['T'], viz_purpose=True)[:, :3]
            else:
                rec = self.predict(inp, labels, filter_transparent=True)
            losses = self.compute_losses(imgs, rec)
            loss_tot.update(losses['total'], N=N)
            loss_rec.update(sum([losses.get(name, 0.) for name in ['rgb', 'perceptual']]), N=N)
            psnr.update(mse2psnr(mse_func(imgs, rec)), N=N)
            ssim.update(1 - ssim_func(imgs, rec).mean(), N=N)
            lpips.update(lpips_func(imgs, rec), N=N)

        return OrderedDict(
            [('n_blocks', n_blocks), ('L_tot', loss_tot.avg), ('L_rec', loss_rec.avg),
             ('PSNR', psnr.avg), ('SSIM', ssim.avg), ('LPIPS', lpips.avg)]
            + [(f'alpha{k}', alpha.item()) for k, alpha in enumerate(opacities)]
        )

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, NV=240):
        path = path or Path('.')
        self.eval()

        # Textures
        if not self.if_bin_render:
            out = path_mkdir(path / 'textures')
            convert_to_img(torch.sigmoid(self.texture_bkg).permute(0, 3, 1, 2)).save(out / 'bkg.png')
            convert_to_img(torch.sigmoid(self.texture_ground).permute(0, 3, 1, 2)).save(out / 'ground.png')
            for k, img in enumerate(torch.sigmoid(self.textures).permute(0, 3, 1, 2)):
                convert_to_img(img).save(out / f'block_{str(k).zfill(2)}.png')

        # Basic 3D
        meshes = self.build_scene(filter_transparent=True, save=True)
        # colors = self.get_scene_face_colors(filter_transparent=True)
        colors = self.get_scene_face_colors(filter_transparent=True, w_env=False)
        save_mesh_as_video(meshes, path / 'rotated_mesh.mp4', renderer=self.renderer)
        save_mesh_as_obj(meshes, path / 'mesh_full.obj')
        clean_mesh = self.build_scene(filter_transparent=True, w_bkg=False, reduce_ground=True)
        save_mesh_as_obj(clean_mesh, path / 'mesh.obj')
        syn_blocks = self.build_blocks(filter_transparent=True, synthetic_colors=True, as_scene=True)
        if len(syn_blocks) == 0:
            return None
        # GT pointcloud
        gt = loader.dataset.pc_gt
        with use_seed(123):
            gt = gt[torch.randperm(len(gt))[:3000]]
        save_ply(path / 'gt.ply', gt)

        # Create renderers
        renderer, renderer_light = self.renderer, self.renderer_light

        # Input specific
        count, N = 0, 10
        R_traj, T_traj = [t.to(device) for t in get_circle_traj(N_views=NV)]
        n_zeros = int(np.log10(N - 1)) + 1
        BS = loader.batch_size
        for j, (inp, labels) in enumerate(loader):
            if count >= N:
                break
            inp = torch_to(inp, device)
            img_src, R_src, T_src = inp['imgs'], inp['R'], inp['T']
            B = min(len(img_src), N - count)
            for k in range(B):
                i = str(j * BS + k).zfill(n_zeros)
                img = img_src[k]
                convert_to_img(img).save(path / f'{i}_inp.png')
                R, T = R_src[k:k + 1], T_src[k:k + 1]
                rec = renderer(meshes, R, T, viz_purpose=True)[:, :3]
                convert_to_img(rec).save(path / f'{i}_rec.png')
                convert_to_img(renderer.draw_edges(rec, syn_blocks, R, T, colors)).save(path / f'{i}_rec_col.png')
                convert_to_img(renderer.draw_edges(img, syn_blocks, R, T, colors)).save(path / f'{i}_rec_col_inp.png')
                rec = renderer_light(syn_blocks, R, T, viz_purpose=True)[:, :3]
                convert_to_img(rec).save(path / f'{i}_rec_syn_nobkg.png')
                rec_wedges = renderer_light.draw_edges(rec, syn_blocks, R, T, linewidth=0.7, colors=(0.3, 0.3, 0.3))
                convert_to_img(rec_wedges).save(path / f'{i}_rec_syn_nobkg_edged.png')
                R, T = R @ R_traj, T.expand(NV, -1)
                save_trajectory_as_video(meshes, path / f'{i}_rec_traj.mp4', R=R, T=T, renderer=renderer)
                save_trajectory_as_video(syn_blocks, path / f'{i}_rec_traj_syn.mp4', R=R, T=T, renderer=renderer_light)
            count += B
