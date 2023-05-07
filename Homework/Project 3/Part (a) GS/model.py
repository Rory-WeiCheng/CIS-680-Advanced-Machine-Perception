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


class SOLO(pl.LightningModule):
    _default_cfg = {
        'num_classes': 4,
        'in_channels': 256,
        'seg_feat_channels': 256,
        'stacked_convs': 7,
        'strides': [8, 8, 16, 32, 32],
        'scale_ranges': [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        'epsilon': 0.2,
        'num_grids': [40, 36, 24, 16, 12],
        'mask_loss_cfg': dict(weight=3),
        'cate_loss_cfg': dict(gamma=2, alpha=0.25, weight=1),
        'postprocess_cfg': dict(cate_thresh=0.2, mask_thresh=0.5, pre_NMS_num=50, keep_instance=5, IoU_thresh=0.5)
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)
        
        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=False, weights_backbone=True)
        self.backbone = pretrained_model.backbone
        ## Catergory branch CNN ##
        self.category_branch = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        ## Mask branch CNN ##
        self.mask_branch = torch.nn.Sequential(
            nn.Conv2d(in_channels=258, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )
        # Depending on the level of the pyramid, the last 
        self.mask_post = nn.ModuleList()
        for grid_size in self.num_grids:
            self.mask_post.append(nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=grid_size**2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()
            ))

    ####################################################################################################
    # Forward function should calculate across each level of the feature pyramid network.
    # Input:
    #     images: batch_size number of images (N, C, H, W)
    # Output:
    #     if eval = False
    #         category_predictions: list, len(fpn_levels), each (batch_size, C-1, S, S)
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, 2*feature_h, 2*feature_w)
    #     if eval==True
    #         category_predictions: list, len(fpn_levels), each (batch_size, S, S, C-1)
    #         / after point_NMS
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, image_h/4, image_w/4)
    #         / after upsampling
    ####################################################################################################
    def forward(self, images, eval=True):
        # Initialize empty container
        category_predictions = []
        mask_predictions = []
        # Generate FPN from pretrained backbone and match stride from [4,8,16,32,64] to [8, 8, 16, 32, 32]
        feature_pyramid = [v.detach() for v in self.backbone(images).values()]
        feature_pyramid[0] = F.interpolate(feature_pyramid[0], (100,136))
        feature_pyramid[-1] = F.interpolate(feature_pyramid[-1], (25,34))
        
        # Iterate through each level in pyramid for category and mask branch
        for ith_level, level in enumerate(feature_pyramid):
            cate_branch_level, mask_branch_level = self.forwardSingleLevel(level, ith_level, eval)
        category_predictions.append(cate_branch_level)
        mask_predictions.append(mask_branch_level)


    def forwardSingleLevel(self, level, ith_level, eval):
        '''
        Input: 
        level: (N,C,H,W) One level from the FPN
        ith_level: (1,) Integer representing the level number
        eval: Boolean variable to determine evaluation output or not
        '''
        pass


    # This function build the ground truth tensor for each batch in the training
    # Input:
    #     bounding_boxes:   list, len(batch_size), each (n_object, 4) (x1 y1 x2 y2 system)
    #     labels:           list, len(batch_size), each (n_object, )
    #     masks:            list, len(batch_size), each (n_object, 800, 1088)
    # Output:
    #     category_targets: list, len(batch_size), list, len(fpn), (S, S), values are {1, 2, 3}
    #     mask_targets:     list, len(batch_size), list, len(fpn), (S^2, 2*feature_h, 2*feature_w)
    #     active_masks:     list, len(batch_size), list, len(fpn), (S^2,)
    #     / boolean array with positive mask predictions
    def generate_targets(self, bounding_boxes, labels, masks):
        # Initialize empty container and bbox resize transform
        category_targets, mask_targets, active_masks = [], [], [] # (N,) (N,) (N,)
        # Iterate through each images in the batch
        for ith_batch in range(len(bounding_boxes)):
            bbox_ith = bounding_boxes[ith_batch]    # (n_object, 4)
            labels_ith = labels[ith_batch]          # (n_object, )
            masks_ith = masks[ith_batch]            # (n_object, 800, 1088)
            category_batch, mask_batch, active_mask_batch = [], [], [] # (5,) (5,) (5,)
            # Iterate through each level in the pyramid
            for ith_level in range(5):
                num_grid = self.num_grids[ith_level]
                stride = self.strides[ith_level]
                category_pyramid_ith = torch.zeros((num_grid,num_grid)) # (S,S)
                h_p, w_p = masks_ith.shape[1] // stride, masks_ith.shape[2] // stride # height and width of current prediction level
                bbox_resize_transform = transforms.Resize((2*h_p, 2*w_p))
                mask_pyramid_ith = torch.zeros((num_grid**2, 2*h_p, 2*w_p)).to(torch.uint8)
                # Find all bbox scale
                for ith_obj in range(len(bbox_ith)):
                    # Original bounding box coordinate
                    x1, y1, x2, y2 = torch.from_numpy(bbox_ith[ith_obj]) # torch.div(torch.from_numpy(bbox_ith[ith_obj]), stride, rounding_mode='floor')
                    ori_w, ori_h = x2 - x1, y2 - y1
                    # If the ith_obj bbox corresponds the ith_level pyramid, then assign it to this level
                    if self.scale_ranges[ith_level][0] <= torch.sqrt(ori_w * ori_h) <= self.scale_ranges[ith_level][1]:
                        # Scaled x1, y1, x2, y2 coordinate of bbox in Pyramid level
                        x1, y1, x2, y2 = torch.from_numpy(bbox_ith[ith_obj]) / stride
                        center_x, center_y = (x1 + x2)/2, (y1 + y2)/2
                        # Find centre region bounding box coordinate
                        sca_w, sca_h = self.epsilon * (x2 - x1), self.epsilon * (y2 - y1)
                        x1_, y1_, x2_, y2_ = center_x - sca_w/2, center_y - sca_h/2, center_x + sca_w/2, center_y + sca_h/2
                        # Find all active grid cell indexes that falls into centre region
                        top_ind = max(0, int(y1_ / h_p * num_grid))
                        bottom_ind = min(num_grid - 1, int(y2_ / h_p * num_grid))
                        left_ind = max(0, int(x1_ / w_p * num_grid))
                        right_ind = min(num_grid - 1, int(x2_ / w_p * num_grid))
                        # Constrain active grid cell region within 3x3
                        center_grid_h = int(center_y / h_p * num_grid)
                        center_grid_w = int(center_x / w_p * num_grid)
                        top = max(top_ind, center_grid_h - 1)
                        bottom = min(bottom_ind, center_grid_h + 1)
                        left = max(left_ind, center_grid_w - 1)
                        right = min(right_ind, center_grid_w + 1)
                        # Assign ith category pyramid level
                        label = labels_ith[ith_obj]
                        x, y = np.meshgrid(range(left, right+1), range(top, bottom+1))
                        category_pyramid_ith[y,x] = label
                        # Assign ith mask pyramid level
                        curr_mask = masks_ith[ith_obj].clone() # (800, 1088)
                        empty_index_region = torch.zeros((num_grid,num_grid)).to(torch.bool)
                        empty_index_region[y,x] = 1
                        channel_index = empty_index_region.flatten()
                        mask_pyramid_ith[channel_index] = bbox_resize_transform(curr_mask.unsqueeze(0)).squeeze().to(torch.uint8)

                active_ith_level = torch.where(category_pyramid_ith != 0, 1, 0).flatten() # (S**2,)
                active_mask_batch.append(active_ith_level.flatten())
                category_batch.append(category_pyramid_ith)
                mask_batch.append(mask_pyramid_ith)
            category_targets.append(category_batch)
            mask_targets.append(mask_batch)
            active_masks.append(active_mask_batch)
        return category_targets, mask_targets, active_masks
