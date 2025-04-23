import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamAutomaticMaskGenerator  # , SamPredictor
from model.build_sam import sam_model_registry
from model.predictor import SamPredictor
import hdbscan
import time

import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pytorch3d.structures import Pointclouds
from mpl_toolkits.mplot3d import Axes3D

import copy

from segment_anything.utils.transforms import ResizeLongestSide

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

sam_checkpoint = "/home/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda:0")
predictor = SamPredictor(sam)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

def SAM_clustering(points, point_idxes, masks, imgs, bsz=1, attn_type=2, is_vis=False):
    block_clusterer = {}
    block_cluster_labels = {}
    block_clusterer_center_coord = {}
    block_bbx = {}
    block_clusterer_center_vert_idx = {}
    block_attn = {}

    for blk_idx in points:
        clusterer = []
        cluster_labels = []
        cluster_center_coord = []
        bbx = []
        cluster_center_idx = []
        Attn = []
        point_batch, mask_batch, point_idx_batch = points[blk_idx], masks[blk_idx], point_idxes[blk_idx]
        for batch_idx, (point, mask, point_idx) in enumerate(zip(point_batch, mask_batch, point_idx_batch)):
            box = preprocessing_mask(imgs[batch_idx], mask)
            bbx.append(box)
            if box is None or point is None:
                cluster = None
            else:
                attn, attn_fin, attn_sum = predict_attn(imgs[batch_idx], point.detach(), box)
                matrix = [attn, attn_fin, attn_sum][attn_type]
                cluster = clustering(matrix, point.detach(), point_idx, viz_2D=is_vis, inp=imgs[batch_idx].cpu(), input_box=box)
                clusterer.append(cluster)
                Attn.append(matrix)
            if cluster is not None:
                cluster_labels.append(cluster.labels_)
                center_coord, center_idx = find_cluster_center_coord(cluster, matrix, point, point_idx)
                cluster_center_coord.append(center_coord)
                cluster_center_idx.append(center_idx)
            else:
                cluster_labels.append(None)
                cluster_center_coord.append(None)
                cluster_center_idx.append(None)
                Attn.append([None])
        block_clusterer[blk_idx] = clusterer
        block_cluster_labels[blk_idx] = cluster_labels
        block_clusterer_center_coord[blk_idx] = cluster_center_coord
        block_bbx[blk_idx] = bbx
        block_clusterer_center_vert_idx[blk_idx] = cluster_center_idx
        block_attn[blk_idx] = Attn

    return block_clusterer, block_cluster_labels, block_clusterer_center_coord, block_bbx, block_clusterer_center_vert_idx, block_attn

def predict_attn_batch(data, input_points, input_box=None, bsz=1):

    device = data.device
    img_size = tuple(data.shape[2:])
    blk_num = len(input_points)
    iter_num = input_points[next(iter(input_points))].shape[1]
    data = np.array(data.cpu().permute(0, 2, 3, 1) * 255).astype(np.uint8)

    batched_input_template = []
    for idx in range(bsz):
        batched_input_template.append({
            'image': prepare_image(data[idx], resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(torch.stack([value[idx] for value in input_box.values()]).to(device), img_size),
            'original_size': img_size,
        })
    # for blk_idx in range(b)
    for idx in range(iter_num):
        batched_input = copy.deepcopy(batched_input_template)
        for b_idx in range(bsz):
            point_coords = torch.stack([value[idx] for value in input_points.values()])[:, b_idx, :]
            point_coords = resize_transform.apply_coords_torch(point_coords, img_size)
            point_labels = torch.tensor([1] * blk_num).to(device)
            point_coords, point_labels = point_coords[None, :, :], point_labels[None, :]

            batched_input[b_idx]['point_coords'] = point_coords
            batched_input[b_idx]['point_labels'] = point_labels
        # if point
        batched_output = sam(batched_input, multimask_output=False)


    return None, None, None


def preprocessing_mask(img=None, mask=None, bsz=1, return_tensor=False):
    if not return_tensor:
        mask = torch.nonzero(mask[0, :, :], as_tuple=True)
        try:
            y_min, x_min = torch.min(mask[0]), torch.min(mask[1])
            y_max, x_max = torch.max(mask[0]), torch.max(mask[1])
            bounding_box = torch.tensor([x_min, y_min, x_max, y_max])
            return np.array(bounding_box.cpu())
        except:
            return None

    bounding_box = torch.zeros((bsz, 4))

    for idx in range(bsz):
        _mask = torch.nonzero(mask[idx, 0, :, :], as_tuple=True)
        y_min, x_min = torch.min(_mask[0]), torch.min(_mask[1])
        y_max, x_max = torch.max(_mask[0]), torch.max(_mask[1])
        bounding_box[idx] = torch.tensor([x_min, y_min, x_max, y_max])


    return bounding_box

def predict_attn(data, input_points, input_box=None, is_attn_vis=False):
    data = np.array(data.cpu().squeeze(0).permute(1, 2, 0) * 255).astype(np.uint8)
    input_label = np.array([1])
    predictor.set_image(data)
    attn = []
    attn_fin = []
    for input_point in input_points:
        input_point = np.array(input_point.unsqueeze(0).cpu())
        attn_ary, attn_fin_ary = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
            vis_attn=False,
            attn_only=True,
        )
        attn_ary = normalize(attn_ary.mean(axis=1).squeeze(0).mean(0))
        attn_fin_ary = normalize(attn_fin_ary.mean(axis=1).squeeze(0).mean(0))
        attn.append(attn_ary)
        attn_fin.append(attn_fin_ary)
        if is_attn_vis:
            vis_array = attn_fin_ary.view(64, 64)
            plt.imshow(vis_array.cpu(), vmin=0, vmax=1)  # Use a grayscale colormap
            plt.axis('off')  # Turn off the axis numbers and ticks
            plt.savefig(f'attn_{input_point}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    attn_sum = [(a + b) / 2. for a, b in zip(attn, attn_fin)]
    return attn, attn_fin, attn_sum

def find_cluster_center_coord(cluster, cluster_data, point, point_idx, is_spilt=False):
    if len(cluster.exemplars_) == 0:
        return -1, -1
    cluster_data = torch.stack([item for item in cluster_data])
    center_coord = []
    center_idx = []
    for label_idx, data in enumerate(cluster.exemplars_):
        mean_vector = torch.tensor(np.mean(data, axis=0)).to(cluster_data.device)
        idx = torch.argmin(torch.norm(cluster_data - mean_vector, dim=1))
        center_coord.append(point[idx.item()])
        center_idx.append(point_idx[idx.item()])
    return center_coord, center_idx


def clustering(cluster_data, input_points=None, point_index=None, block=None, viz_2D=False, viz_3D=False, inp=None, input_box=None, clusterer = None, new_data=None):
    if new_data:
        assert clusterer is not None
        new_data = np.stack([tensor.cpu().numpy() for tensor in new_data])

        new_labels, _ = hdbscan.approximate_predict(clusterer, new_data)
        return
    else:
        try:
            data_matrix = torch.stack([item.cpu() for item in cluster_data]).numpy()
        except:
            return None
        # data_matrix = pre_prcessing(data_matrix)
        # Perform HDBSCAN clustering
        start_time = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2, cluster_selection_method='eom')#, prediction_data=True)#, allow_single_cluster=True)
        try:
            clusterer = clusterer.fit(data_matrix)
        except:
            return None
        cluster_labels = clusterer.labels_
        end_time = time.time()
        # print(f"Clustering took {end_time - start_time} seconds.")

    if viz_2D:
        assert input_points is not None
        data = {
            'X': input_points[:, 0].cpu(),  # X coordinates
            'Y': input_points[:, 1].cpu(),  # Y coordinates
            'Cluster': cluster_labels,  # Cluster labels
            'Index': point_index.cpu()
        }

        # Create a DataFrame

        df = pd.DataFrame(data)
        img = np.array(inp.squeeze(0).permute(1, 2, 0) * 255).astype(np.uint8)

        # Plot
        plt.figure(figsize=(5, 4))
        plt.imshow(img)
        if input_box is not None:
            show_box(input_box, plt.gca())
        sns.scatterplot(data=df, x='X', y='Y', hue='Cluster', palette='bright', legend='full', style='Cluster')
        plt.title('Clustering Results')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        plt.xlim(0, 400)
        plt.ylim(300, 0)

        for i, point in df.iterrows():
            plt.text(point['X'] + 0.02,  # X location plus a little offset
                     point['Y'],
                     point['Index'],
                     fontsize=9)

        plt.show()
    if viz_3D:
        assert block is not None
        vertices_list = block.verts_list()
        all_labels = -np.ones(vertices_list[0].shape[0], dtype=int)  # Use -1 for points not part of the clustering
        all_labels[input_points] = cluster_labels
        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Unique cluster labels, including -1 if present
        unique_labels = set(all_labels)
        colors = plt.cm.Spectral(
            np.linspace(0, 1, len(unique_labels) - (-1 in unique_labels)))  # Adjust color map if -1 is present

        color_index = 0
        for k in unique_labels:
            if k == -1:
                col = 'k'  # black color for noise
                label = 'Noise'
            else:
                col = colors[color_index]
                label = f'Cluster {k}'
                color_index += 1

            class_member_mask = (all_labels == k)

            xyz = vertices_list[0][class_member_mask].cpu()
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=20, color=col, label=label)

        ax.set_xlim([0.5, -0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])

        ax.set_title('3D Clustering Results with HDBSCAN')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.legend()
        ax.grid(True)
        # ax.scatter(0, 0, 0, color="r", s=100)

        plt.show()
    return clusterer

def pre_prcessing(data_matrix):
    # Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    pca = PCA(n_components=data_matrix.shape[0])
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data

def normalize(tensor):
    assert type(tensor) is torch.Tensor
    normalized_tensors = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    return normalized_tensors

def unify_dict(dict_list):
    res = []
    for tensor_dict in dict_list:
        for key, tensor in tensor_dict.items():
            if tensor.dim() > 1:
                tensor_dict[key] = tensor.mean(dim=0)
            else:
                continue
        res.append(tensor_dict)
    return res

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))