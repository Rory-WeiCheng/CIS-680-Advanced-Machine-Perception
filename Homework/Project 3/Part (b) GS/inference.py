import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader

try:
    import dataset
    import model
    import train
except ModuleNotFoundError:
    pass

def visualTrainingCurve():
    total_loss = np.load('result/total_loss.npy',allow_pickle=True)
    cate_loss = np.load('result/cate_loss.npy',allow_pickle=True)
    mask_loss = np.load('result/mask_loss.npy',allow_pickle=True)
    valid_loss = np.load('result/valid_loss.npy',allow_pickle=True)
    # Visualize training curve
    fig, ax =plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(total_loss)
    ax[0].set_title('Raw training loss')
    avg_total_loss = []
    step = 5
    for i in range(0, len(total_loss), step):
        avg_total_loss.append(np.mean(total_loss[i:i*step]))
    ax[1].plot(avg_total_loss)
    ax[1].set_title('Smoothed training loss')
    # Visualize category loss
    fig, ax =plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(cate_loss)
    ax[0].set_title('Raw category loss')
    avg_total_loss = []
    step = 5
    for i in range(0, len(cate_loss), step):
        avg_total_loss.append(np.mean(cate_loss[i:i*step]))
    ax[1].plot(avg_total_loss)
    ax[1].set_title('Smoothed category loss')
    # Visualize mask loss
    fig, ax =plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(mask_loss)
    ax[0].set_title('Raw mask loss')
    avg_total_loss = []
    step = 5
    for i in range(0, len(mask_loss), step):
        avg_total_loss.append(np.mean(mask_loss[i:i*step]))
    ax[1].plot(avg_total_loss)
    ax[1].set_title('Smoothed mask loss')
    # Visualize validation loss
    valid_loss = valid_loss[:9350]
    fig, ax =plt.subplots(1,2,figsize=(12,5))
    ax[0].plot(valid_loss)
    ax[0].set_title('Raw valid loss')
    avg_total_loss = []
    step = 10
    for i in range(0, len(valid_loss), step):
        avg_total_loss.append(np.mean(valid_loss[i:i*step]))
    ax[1].plot(avg_total_loss)
    ax[1].set_title('Smoothed valid loss')


def visualDataLoading(solo_dataset):
    untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
    color = [np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0])] # Label: 0 - Vehicle: Blue, 1 - Human: Green, 2 - Animal: Red

    for img_idx in [5,1,28,4,45]: # [5,1,28,4,45]
        img_ith, label_ith, mask_ith, bbox_ith = solo_dataset[img_idx]
        # Restore image from normalized dataset
        img_ith = untransform(img_ith).numpy().astype(np.float64).transpose(1,2,0) # (800, 1088, 3)
        img_ith[:,:11,:] = 0
        img_ith[:,-11:,:] = 0
        disolve_img = img_ith.copy()
        img_copy = img_ith.copy()
        for mask_idx in range(mask_ith.shape[0]):
            bool_map = mask_ith[mask_idx] > 0
            disolve_img[bool_map] = color[label_ith[mask_idx] - 1]
        final_img = cv2.addWeighted(disolve_img, 0.5, img_copy, 0.5,0)
        # Plot the bounding box
        plt.figure(figsize=(7,6))
        plt.imshow(final_img)
        for bbox_idx in range(bbox_ith.shape[0]):
            top_x, top_y = bbox_ith[bbox_idx][0], bbox_ith[bbox_idx][1]
            width, height = bbox_ith[bbox_idx][2] - bbox_ith[bbox_idx][0], bbox_ith[bbox_idx][3] - bbox_ith[bbox_idx][1]
            plt.gca().add_patch(Rectangle((top_x, top_y), width, height, linewidth=1, edgecolor='r',facecolor='none'))
        plt.show()


def visual_target_building(img, category_targets_ith, mask_targets_ith, active_masks_ith):
    '''
    Input:
    img:              (3, 800, 1088)
    category_targets: list, len(fpn), (S, S), values are {1, 2, 3}
    mask_targets:     list, len(fpn), (S^2, 2*feature_h, 2*feature_w)
    active_masks:     list, len(fpn), (S^2,)
    '''
    mask_untransform = transforms.Resize((800, 1088))
    img_untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
    color = [np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0])]

    plt.figure(figsize=(40,10))
    for level_ith in range(5):
        img_ith = img_untransform(img).numpy().astype(np.float64).transpose(1,2,0) # (800, 1088, 3)
        img_ith[:,:11,:] = 0
        img_ith[:,-11:,:] = 0
        cate_ith_level = category_targets_ith[level_ith]
        mask_ith_level = mask_targets_ith[level_ith] 
        active_mask_level = active_masks_ith[level_ith].to(torch.bool)
        num_grid = cate_ith_level.shape[0]
        # If current level has active grid cell
        if torch.sum(active_mask_level) > 0:
            mask_all = mask_untransform(mask_ith_level[active_mask_level]).numpy()
            active_index = torch.nonzero(active_mask_level).numpy()
            for i in range(mask_all.shape[0]):
                row_idx, col_idx = active_index[i] // num_grid, active_index[i] % num_grid
                label = int(cate_ith_level[row_idx, col_idx].item())
                img_ith = np.where(mask_all[i][...,None], color[label - 1], img_ith) 
        plt.subplot(1,5, level_ith+1)
        plt.title(f'P{level_ith+2} level')
        plt.imshow(img_ith)


def visualTargetBuilding(solo, solo_dataset):
    for idx in [5,1,28,4,45]:
        sample_img, sample_label, sample_mask, sample_bbox = solo_dataset[idx]
        category_targets, mask_targets, active_masks, _, _ = solo.generate_targets(sample_bbox.unsqueeze(0), sample_label.unsqueeze(0), sample_mask.unsqueeze(0))
        visual_target_building(sample_img, category_targets[0], mask_targets[0], active_masks[0])


def visualGT(solo_dataset, img_idx):
    untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
    color = [np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0])] # Label: 0 - Vehicle: Blue, 1 - Human: Green, 2 - Animal: Red
    img_ith, label_ith, mask_ith, bbox_ith = solo_dataset[img_idx]
    # Restore image from normalized dataset
    img_ith = untransform(img_ith).numpy().astype(np.float64).transpose(1,2,0) # (800, 1088, 3)
    img_ith[:,:11,:] = 0
    img_ith[:,-11:,:] = 0
    disolve_img = img_ith.copy()
    img_copy = img_ith.copy()
    for mask_idx in range(mask_ith.shape[0]):
        bool_map = mask_ith[mask_idx] > 0
        disolve_img[bool_map] = color[label_ith[mask_idx] - 1]
    final_img = cv2.addWeighted(disolve_img, 0.5, img_copy, 0.5,0)
    # Plot the bounding box
    plt.figure(figsize=(5,5))
    plt.imshow(final_img)


def visualInferenceResult(sample_img, mask_batch, cate_batch):
    # Visualize inference mask with image
    untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
    color = [np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0])]

    img_ith = untransform(sample_img).numpy().astype(np.float64).transpose(1,2,0) # (800, 1088, 3)
    img_ith[:,:11,:] = 0
    img_ith[:,-11:,:] = 0

    mask_sample = mask_batch[0].cpu().detach().numpy()
    label = cate_batch[0].cpu().detach().numpy()

    plt.figure(figsize=(40,10))
    for i in range(5):
        disolve_img = img_ith.copy()
        plt.subplot(1,5,i+1)
        bool_map = mask_sample[i] > 0.2
        disolve_img[bool_map] = color[label[i]]
        masked_img = cv2.addWeighted(disolve_img, 0.5, img_ith, 0.5,0)
        plt.imshow(masked_img)


def visualInferenceResultTotal(solo, solo_dataset):
    # idx = 2024 # 40, 24, 22, 12, 56, 456, 2022, 2010
    for idx in [258, 456, 302, 47, 1827]:
        sample_img, sample_label, sample_mask, sample_bbox = solo_dataset[idx]
        # category_predictions, mask_predictions = solo.forward(sample_img.unsqueeze(0), eval=False)
        mask_batch, score_batch, cate_batch = solo.forward(sample_img.unsqueeze(0).cuda(), eval=True)
        # Visualize image with GT mask
        visualGT(solo_dataset, idx)
        visualInferenceResult(sample_img, mask_batch, cate_batch)
        del(sample_img, sample_label, sample_mask, sample_bbox, mask_batch, score_batch, cate_batch)


if __name__ == '__main__':
    solo_dataset = dataset.SOLO_Dataset(img_path='data/hw3_mycocodata_img_comp_zlib.h5',
                        mask_path='data/hw3_mycocodata_mask_comp_zlib.h5',
                        bbox_path='data/hw3_mycocodata_bboxes_comp_zlib.npy',
                        label_path='data/hw3_mycocodata_labels_comp_zlib.npy')

    visualDataLoading(solo_dataset)