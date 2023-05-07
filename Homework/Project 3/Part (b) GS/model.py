import numpy as np
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as transforms


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
            
        self.automatic_optimization=False
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

        self.cate_loss = []
        self.mask_loss = []
        self.total_loss = []
        self.val_loss = []

    def forward(self, images, eval=False):
        '''
        Forward function should calculate across each level of the feature pyramid network.
        Input:
            images: batch_size number of images (N, C, H, W)
        Output:
            if eval = False
                category_predictions: list, len(fpn_levels), each (batch_size, C-1, S, S)
                mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, 2*feature_h, 2*feature_w)
            if eval==True
                category_predictions: list, len(fpn_levels), each (batch_size, S, S, C-1)
                / after point_NMS
                mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, image_h/4, image_w/4)
                / after upsampling
        '''
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

        if not eval:
            return category_predictions, mask_predictions
        else:
            b, _, h, w = images.shape
            category_predictions = [self.points_nms(category_predictions_level_ith) for category_predictions_level_ith in category_predictions]
            # Concatenate category prediction to be (batch_size, sum of S**2, C)
            cat_pred_concat = torch.cat([level_ith.flatten(2).permute(0,2,1) for level_ith in category_predictions], dim=1)
            # Concatenate mask prediction to be (batch_size, sum of S**2, H/4, W/4) where H,W is original image size
            mask_pred_concat = torch.cat([F.interpolate(level_ith, size=(h//4, w//4)) for level_ith in mask_predictions], dim=1)
            # Iterate through each images in the batch for post-processing
            mask_batch, score_batch, cate_batch = [], [], []
            for batch_idx in range(b):
                cate_pred_ith = cat_pred_concat[batch_idx]  # (N, 3)
                mask_pred_ith = mask_pred_concat[batch_idx] # (N, h/4, w/4)
                # Get all grid cells with c_max > threshold
                cate_thresh = self.postprocess_cfg['cate_thresh']
                cat_max_all, cat_max_index = torch.max(cate_pred_ith, dim=-1)
                cat_max_score_thresh = cat_max_all[cat_max_all > cate_thresh]      # (n,)
                cat_max_index_thresh = cat_max_index[cat_max_all > cate_thresh]    # (n,)
                mask_pred_thresh = mask_pred_ith[cat_max_all > cate_thresh]        # (n,h/4,w/4)
                # Get score for all threshed grid cell
                mask_thresh = self.postprocess_cfg['mask_thresh']
                nom = torch.sum(mask_pred_thresh * (mask_pred_thresh > mask_thresh), dim=(1,2))
                den = torch.sum((mask_pred_thresh > mask_thresh), dim=(1,2)) + 1e-5
                cell_thresh_score = nom / den
                cell_score = cell_thresh_score * cat_max_score_thresh
                # Get sorted score and mask to perform matrixNMS
                sorted_score, sorted_index = torch.sort(cell_score, descending=True)
                sorted_mask = mask_pred_thresh[sorted_index]
                sorted_cate = cat_max_index_thresh[sorted_index]
                # Get NMS score and select highest k masks
                nms_score = self.MatrixNMS(sorted_mask, sorted_score, method='non_Gaussian')
                _, nms_sorted_index = torch.sort(nms_score, descending=True)
                k_highest_index = nms_sorted_index[:self.postprocess_cfg['keep_instance']]
                final_mask_pre = sorted_mask[k_highest_index] # (k, h/4, w/4)
                final_mask = F.interpolate(final_mask_pre.unsqueeze(0), (h,w))
                final_score = nms_score[k_highest_index]  # (k,)
                final_cate = sorted_cate[k_highest_index] # (k,)
                # Append current image mask, cate, score
                mask_batch.append(final_mask.squeeze())
                score_batch.append(final_score)
                cate_batch.append(final_cate)
            return mask_batch, score_batch, cate_batch
            


    def forwardSingleLevel(self, level, ith_level, eval):
        '''
        Input: 
        level: (N,C,H,W) One level from the FPN
        ith_level: (1,) Integer representing the level number
        eval: Boolean variable to determine evaluation output or not
        Output:
        cate_branch_level: (N, C, S, S)
        mask_branch_level: (N, S**2, 2*H, 2*W)
        '''
        # Forward single level for category branch
        num_grid = self.num_grids[ith_level]
        aligned_level = F.interpolate(level, (num_grid,num_grid), mode='bilinear')
        cate_branch_level = self.category_branch(aligned_level)

        # Forward single level for mask branch
        # 1. Create two (x,y) channels
        N, C, H, W = level.shape
        y, x = torch.meshgrid(torch.arange(0,H), torch.arange(0,W), indexing='ij')
        y_norm, x_norm = y/(H-1) * 2 - 1, x/(W-1) * 2 - 1 # (H,W)
        # 2. Concatenate them after C channels
        yx_norm = torch.stack((y_norm, x_norm))
        yx_norm_repeat = yx_norm.repeat(N,1,1,1)
        coord_level = torch.cat((level, yx_norm_repeat.to(level.device)), dim=1)
        # 3. Feed into FCN for mask branch
        mask_branch_level = self.mask_branch(coord_level)
        # 4. Resize to (2*H, 2*W) and feed into last layer
        mask_branch_level = F.interpolate(mask_branch_level, (2*H,2*W))
        mask_branch_level = self.mask_post[ith_level](mask_branch_level)

        return cate_branch_level, mask_branch_level


    def generate_targets(self, bounding_boxes, labels, masks):
        '''
        This function build the ground truth tensor for each batch in the training
        Input:
            bounding_boxes:   list, len(batch_size), each (n_object, 4) (x1 y1 x2 y2 system)
            labels:           list, len(batch_size), each (n_object, )
            masks:            list, len(batch_size), each (n_object, 800, 1088)
        Output:
            category_targets: list, len(batch_size), list, len(fpn), (S, S), values are {1, 2, 3}
            mask_targets:     list, len(batch_size), list, len(fpn), (S^2, 2*feature_h, 2*feature_w)
            active_masks:     list, len(batch_size), list, len(fpn), (S^2,)
            / boolean array with positive mask predictions
        '''
        # Initialize empty container and bbox resize transform
        category_targets, mask_targets, active_masks = [], [], [] # (N,) (N,) (N,)
        mask_targets_sub = [[] for _ in range(5)]
        active_masks_sub = [[]for _ in range(5)]
        H, W = masks[0].shape[-2:]
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
                category_pyramid_ith = torch.zeros((num_grid,num_grid), dtype=torch.uint8) # (S,S)
                h_p, w_p = H // stride, W // stride # h_p, w_p = masks_ith.shape[1] // stride, masks_ith.shape[2] // stride
                bbox_resize_transform = transforms.Resize((2*h_p, 2*w_p))
                mask_pyramid_ith = torch.zeros((num_grid**2, 2*h_p, 2*w_p)).to(torch.uint8)
                # Find all bbox scale
                for ith_obj in range(len(bbox_ith)):
                    # Original bounding box coordinate
                    x1, y1, x2, y2 = bbox_ith[ith_obj] # x1, y1, x2, y2 = torch.from_numpy(bbox_ith[ith_obj])
                    ori_w, ori_h = x2 - x1, y2 - y1
                    # If the ith_obj bbox corresponds the ith_level pyramid, then assign it to this level
                    if self.scale_ranges[ith_level][0] <= torch.sqrt(ori_w * ori_h) <= self.scale_ranges[ith_level][1]:
                        # Scaled x1, y1, x2, y2 coordinate of bbox in Pyramid level
                        x1, y1, x2, y2 = bbox_ith[ith_obj] / stride # x1, y1, x2, y2 = torch.from_numpy(bbox_ith[ith_obj]) / stride
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
                        # Assign ith category and mask pyramid level for ith mask in ith batch image
                        label = labels_ith[ith_obj].to(torch.uint8)
                        transformed_mask = bbox_resize_transform(masks_ith[ith_obj].unsqueeze(0)).squeeze().to(torch.uint8)
                        for y in range(top, bottom + 1):
                            for x in range(left, right + 1):
                                category_pyramid_ith[y,x] = label
                                mask_pyramid_ith[y*num_grid + x] = transformed_mask
                                mask_targets_sub[ith_level].append(transformed_mask)
                                active_masks_sub[ith_level].append(torch.Tensor([ith_batch, y*num_grid + x]))
                active_ith_level = torch.where(category_pyramid_ith != 0, 1, 0).flatten() # (S**2,)
                active_mask_batch.append(active_ith_level.flatten())
                category_batch.append(category_pyramid_ith)
                mask_batch.append(mask_pyramid_ith)
            category_targets.append(category_batch)
            mask_targets.append(mask_batch)
            active_masks.append(active_mask_batch)
        return category_targets, mask_targets, active_masks, mask_targets_sub, active_masks_sub


    def loss(self, category_predictions, mask_predictions, category_targets, mask_targets_sub, active_masks_sub):
        '''
        Input:
        category_predictions: list, len(fpn_levels), each (batch_size, C-1, S, S)
        mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, 2*feature_h, 2*feature_w)
        category_targets:     list, len(batch_size), list, len(fpn), (S, S), values are {1, 2, 3}
        mask_targets_sub:     list, len(fpn), list, (N,) each (2*feature_h, 2*feature_w)
        active_masks_sub:     list, len(fpn), list, (N,) each (2,)

        Output:
        loss: category_loss + lambda * mask_loss
        '''
        # Slice ground truth category to five levels, where each level is (batch_size, S, S) S->{40,36,24,16,12}
        cate_gt_l1, cate_gt_l2, cate_gt_l3, cate_gt_l4, cate_gt_l5 = map(list, zip(*category_targets))
        cate_gt_l1, cate_gt_l2, cate_gt_l3, cate_gt_l4, cate_gt_l5 = torch.stack(cate_gt_l1), torch.stack(cate_gt_l2), torch.stack(cate_gt_l3), torch.stack(cate_gt_l4), torch.stack(cate_gt_l5)
        cate_gt_list = [cate_gt_l1, cate_gt_l2, cate_gt_l3, cate_gt_l4, cate_gt_l5]
        
        # Calculate the category loss
        cate_loss = self.focal_loss(category_predictions, cate_gt_list)
        # Calculate the mask loss
        mask_loss = self.dice_loss(mask_predictions, mask_targets_sub, active_masks_sub)

        # total loss
        L = self.cate_loss_cfg["weight"] * cate_loss + self.mask_loss_cfg["weight"] * mask_loss
        
        return cate_loss, mask_loss, L


    def focal_loss(self, category_predictions, cate_gt_list):
        '''
        Calculate the focal loss based on category prediction and gt
        '''
        fc_loss = 0
        # Weight factor for category loss
        alpha = self.cate_loss_cfg["alpha"]
        gamma = self.cate_loss_cfg["gamma"]
        # Calculate the category loss
        for level_ith in range(5):
            cate_pred_ith = category_predictions[level_ith] # (bs,C-1,S,S)
            bs, C, S, _ = cate_pred_ith.shape
            cate_gt_ith = cate_gt_list[level_ith].to(cate_pred_ith.device) # (bs,S,S)

            # Get flattened prediction tensor and gt tensor
            p = cate_pred_ith.flatten(start_dim=2).permute(0,2,1).flatten() # from (bz,C,S,S) -> (1,bz*C*S*S)
            cate_gt_ith_flatten = cate_gt_ith.flatten().to(torch.long)
            cate_gt_ith_shape = torch.zeros((bs*S*S, 3))
            row_index = range(bs*S*S)
            cate_gt_ith_shape[row_index,cate_gt_ith_flatten - 1] = 1
            y = cate_gt_ith_shape.flatten().to(cate_pred_ith.device) # to device?

            fl_1 = -alpha * torch.mean(torch.pow(1 - p, gamma) * torch.log(p + 1e-5) * y)
            fl_2 = -(1-alpha) * torch.mean(torch.pow(p, gamma) * torch.log(1 - p + 1e-5) * (1 - y))
            fc_loss += (fl_1 + fl_2)
        return fc_loss


    def dice_loss(self, mask_predictions, mask_targets_sub, active_masks_sub):
        '''
        Calculate the dice loss 
        '''
        dice_loss = 0
        # Iterate through all levels in FPN and only check active grid cells
        for level_ith in range(5):
            mask_pred_ith = mask_predictions[level_ith] # (bz, S**2, 2H, 2W)
            mask_gt_ith = mask_targets_sub[level_ith]
            active_mask_gt_ith = active_masks_sub[level_ith]
            # If the current level has any active grid cell
            if len(mask_gt_ith) != 0:
                mask_gt_ith = torch.stack(mask_gt_ith) # (N, 2H, 2W)
                active_mask_gt_ith = torch.stack(active_mask_gt_ith).to(torch.long) # (N, 2)
                mask_pred_corr = torch.stack([mask_pred_ith[ith_batch[0], ith_batch[1]] for ith_batch in active_mask_gt_ith]) # (N, 2H, 2W)
                D_mask = 2 * torch.sum(mask_gt_ith * mask_pred_corr, (1,2)) / (torch.sum(torch.pow(mask_gt_ith,2),(1,2)) + torch.sum(torch.pow(mask_pred_corr,2),(1,2)))
                d_mask = 1 - D_mask
                L_mask = torch.mean(d_mask)
                dice_loss += L_mask
        return dice_loss


    def points_nms(self, heat, kernel=2):
        hmax = F.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep


    def MatrixNMS(self, sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
        n = len(sorted_scores)
        sorted_masks = sorted_masks.reshape(n, -1)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)

        ious_cmax = ious.max(0)[0].expand(n, n).T
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        return sorted_scores * decay


    #####################
    # Pytorch-Lightning #
    #####################
    def training_step(self, batch, batch_idx):
        img, labels, masks, bbox  = batch
        category_predictions, mask_predictions = self.forward(img)
        category_targets, _, _, mask_targets_sub, active_masks_sub = self.generate_targets(bbox, labels, masks)
        opt = self.optimizers()
        opt.zero_grad()
        loss1, loss2, loss3 = self.loss(category_predictions, mask_predictions, category_targets, mask_targets_sub, active_masks_sub)
        self.log("train_loss", loss3)
        self.total_loss.append(loss3.item())
        self.cate_loss.append(loss1.item())
        self.mask_loss.append(loss2.item())
        # self.trainer.train_loop.running_loss.append(loss3)
        self.manual_backward(loss3)
        opt.step()
        return loss3


    def on_train_epoch_start(self):
        if self.current_epoch == 26 or self.current_epoch == 32:
            for g in self.optimizers().optimizer.param_groups:
                g['lr'] *= 0.1


    def validation_step(self, batch, batch_idx):
        img, labels, masks, bbox  = batch
        category_predictions, mask_predictions = self.forward(img)
        category_targets, _, _, mask_targets_sub, active_masks_sub = self.generate_targets(bbox, labels, masks)
        _, _, loss = self.loss(category_predictions, mask_predictions, category_targets, mask_targets_sub, active_masks_sub)
        self.log("valid_loss", loss)
        self.val_loss.append(loss.item())
    

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        return optimizer
