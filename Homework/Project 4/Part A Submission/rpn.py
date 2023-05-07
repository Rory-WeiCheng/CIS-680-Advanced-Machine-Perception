import torch
from dataset import *
from utils import *
from torchvision import transforms, ops
from torch import nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import matplotlib.patches as patches


class RPNHead(pl.LightningModule):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=0.8,scale=256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()
        # self.device = device
        # Define Backbone
        self.backbone = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Define Intermediate Layer
        self.inter = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Define Proposal Classifier Head
        self.cls_head = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )
        # Define Proposal Regressor Head
        self.reg_head = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1, padding='same')
        )

        #  find anchors
        self.anchors_param = anchors_param
        self.anchors = self.create_anchors(self.anchors_param['ratio'], 
                                           self.anchors_param['scale'], 
                                           self.anchors_param['grid_size'], 
                                           self.anchors_param['stride']) # (grid_size[0], grid_size[1], 4)
        self.ground_dict = {}
        
        # Initialize empty container to store training and validation loss
        self.total_loss = []
        self.cls_loss = []
        self.reg_loss = []
        self.val_loss = []


    def forward(self, x):
        '''
        Forward the input through the backbone the intermediate layer and the RPN heads
        Input:
            X: (bz, 3, image_size[0],image_size[1])}
        Ouput:
            logits: (bz, 1, grid_size[0],grid_size[1])}
            bbox_regs: (bz, 4, grid_size[0], grid_size[1])}
        '''
        # Forward through the Backbone
        x = self.backbone(x)

        # Forward through the Intermediate layer
        x = self.inter(x)

        # Forward through the Classifier Head
        logits = self.cls_head(x)

        # Forward through the Regressor Head
        bbox_regs = self.reg_head(x)

        # assert logits.shape[1:4] == (1,self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])
        # assert bbox_regs.shape[1:4] == (4,self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])
        return logits, bbox_regs


    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        '''
        This function creates the anchor boxes
        Output:
            anchors: (grid_size[0], grid_size[1], 4)
        '''
        # Find anchor box height and width
        h = scale / np.sqrt(aspect_ratio)
        w = aspect_ratio * h
        # Create center x and y coordinates for all grid cells
        grid_y_range, grid_x_range = grid_sizes
        y, x = torch.meshgrid(torch.arange(grid_y_range), torch.arange(grid_x_range), indexing='ij')
        center_y, center_x = y + 0.5, x + 0.5
        # Append (cx, cy, w, h) into anchors
        anchors = torch.zeros(grid_y_range, grid_x_range, 4)
        anchors[:,:,0] = center_x * stride
        anchors[:,:,1] = center_y * stride
        anchors[:,:,2] = w
        anchors[:,:,3] = h
        return anchors


    def get_anchors(self):
        return self.anchors


    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        '''
        This function creates the ground truth for a batch of images by using
        create_ground_truth internally
        Input:
            bboxes_list:  list:len(bz)  (n_obj,4)
            indexes:      list:len(bz)  (1,)
            image_shape:  tuple:len(2)  (800, 1088)
        Output:
            ground_class: (bz,1,grid_size[0],grid_size[1])
            ground_coord: (bz,4,grid_size[0],grid_size[1])
        '''
        # Initialize empty container
        ground_class, ground_coord = [], []
        # Iterate throgh all batches to find the ground_class and ground_coord
        for batch_idx in range(len(bboxes_list)):
            bbox_ith = bboxes_list[batch_idx]
            index_ith = indexes[batch_idx]
            ground_class_idx, ground_coord_idx = self.create_ground_truth(bbox_ith, 
                                                                          index_ith, 
                                                                          self.anchors_param['grid_size'], 
                                                                          self.get_anchors().to(bbox_ith.device), image_shape)
            ground_class.append(ground_class_idx)
            ground_coord.append(ground_coord_idx)
        # Reshape to specified dimension
        ground_class = torch.stack(ground_class)
        ground_coord = torch.stack(ground_coord)
        return ground_class, ground_coord


    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        '''
        This function creates the ground truth for one image
        It also caches the ground truth for the image using its index
        Input:
                bboxes:      (n_boxes, 4)
                index:       scalar (the index of the image in the total dataset used for caching)
                grid_size:   tuple:len(2)
                anchors:     (grid_size[0], grid_size[1], 4)
        Output:
                ground_clas:  (1, grid_size[0], grid_size[1])
                ground_coord: (4, grid_size[0], grid_size[1])
        '''
        # If pre-calculated image is called then return cached result
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord
        # Otherwise, calculate the ground truth class and coord for new index image
        image_H, image_W = image_size
        ground_class = torch.zeros(grid_size).to(bboxes.device)
        ground_coord = torch.zeros((4, grid_size[0] * grid_size[1])).to(bboxes.device)

        # 1. Find bbox within boundary grid cell mask
        bbox_x_min = anchors[:,:,0] - anchors[:,:,2] / 2
        bbox_x_max = anchors[:,:,0] + anchors[:,:,2] / 2
        bbox_y_min = anchors[:,:,1] - anchors[:,:,3] / 2
        bbox_y_max = anchors[:,:,1] + anchors[:,:,3] / 2
        x_valid = torch.logical_and(bbox_x_min >=0, bbox_x_max < image_W)
        y_valid = torch.logical_and(bbox_y_min >=0, bbox_y_max < image_H)
        boundary_valid_mask = torch.logical_and(x_valid, y_valid) # (50, 68)

        # 2. Calculate cross IOU between all anchor boxes and ground truth boxes
        anchors_flatten = anchors.clone().view(-1,4)
        gt_boxes_cxcywh = ops.box_convert(bboxes, 'xyxy','cxcywh') # (n_obj, 4)
        cross_IOU = calc_IOU(anchors_flatten, gt_boxes_cxcywh) # (50*68, n_obj)

        # 3. Based on IOU, find positive mask and negative mask
        # Position condition 1: IOU > 0.7 with any of the ground truth bbox
        thresh_mask = (cross_IOU > 0.7).any(dim=1)
        # Positive condition 2: highest IOU with a ground truth box
        value, _ = torch.max(cross_IOU, 0)
        for i in range(len(value)):
            thresh_mask = torch.logical_or(thresh_mask, cross_IOU[:,i] == value[i])
        # Get positive anchor mask
        positive_mask = boundary_valid_mask * thresh_mask.view(grid_size)
        # Find negative mask
        negative_mask = torch.logical_not(positive_mask)
        thresh_mask = (cross_IOU < 0.3).all(dim=1)
        negative_mask = boundary_valid_mask * negative_mask * thresh_mask.view(grid_size)

        # 4. Construct the ground class label (1 for positive, -1 for negative, 0 for ignore)
        ground_class[positive_mask] = 1
        ground_class[negative_mask] = -1
        ground_class = ground_class.unsqueeze(0)

        # 5. Construct the ground truth coordinate for all positive cells
        pos_cell_mask_flatten = positive_mask.flatten() # (3400,)
        assigned_gt_index = torch.argmax(cross_IOU, dim=1)[pos_cell_mask_flatten] # (N_pos,)
        pos_cell_bbox = anchors_flatten[pos_cell_mask_flatten] # (N_pos, 4)
        ground_coord[0, pos_cell_mask_flatten] = (gt_boxes_cxcywh[assigned_gt_index][:,0] - pos_cell_bbox[:,0]) / pos_cell_bbox[:,2]
        ground_coord[1, pos_cell_mask_flatten] = (gt_boxes_cxcywh[assigned_gt_index][:,1] - pos_cell_bbox[:,1]) / pos_cell_bbox[:,3]
        ground_coord[2, pos_cell_mask_flatten] = torch.log(gt_boxes_cxcywh[assigned_gt_index][:,2] / pos_cell_bbox[:,2])
        ground_coord[3, pos_cell_mask_flatten] = torch.log(gt_boxes_cxcywh[assigned_gt_index][:,3] / pos_cell_bbox[:,3])
        ground_coord = ground_coord.view(4, grid_size[0], grid_size[1])
        
        # Cache current image ground truth information
        self.ground_dict[key] = (ground_class, ground_coord)
        return ground_class, ground_coord


    def loss_class(self, p_out, n_out):
        '''
        Compute the loss of the classifier
        Input:
            p_out: (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
            n_out: (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels)
        '''
        # Compute classifier's loss
        p_gt = torch.ones_like(p_out)
        n_gt = torch.zeros_like(n_out)
        BCEloss = torch.nn.BCELoss(reduction='mean')
        avg_cls_loss = BCEloss(p_out, p_gt) + BCEloss(n_out, n_gt)
        sum_count = len(p_out) + len(n_out)
        return avg_cls_loss, sum_count


    def loss_reg(self, pos_target_coord, pos_out_r):
        '''
        Compute the loss of the regressor
        Input:
            pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors 
                                                          with positive gt labels)
            pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
        '''
        # Compute regressor's loss
        sum_count = len(pos_target_coord)
        smooth_L1 = torch.nn.SmoothL1Loss(reduction='sum')
        loss = smooth_L1(pos_out_r.flatten(), pos_target_coord.flatten())
        avg_reg_loss = loss / sum_count
        return avg_reg_loss, sum_count


    def compute_loss(self, pred_cls, pred_reg, targ_cls, targ_reg, l=1, effective_batch=50):
        '''
        Compute the total loss
        Input:
            pred_cls: (bz,1,grid_size[0],grid_size[1])
            pred_reg: (bz,4,grid_size[0],grid_size[1])
            targ_cls: (bz,1,grid_size[0],grid_size[1])
            targ_reg: (bz,4,grid_size[0],grid_size[1])
            l: lambda constant to weight between the two losses
            effective_batch: the number of anchors in the effective batch (M in the handout)
        '''
        # Reshape cls from (bz, 1, Sy, Sx) to (bz * Sy * Sx,) and reg from (bz, 4, Sy, Sx) to (bz * Sy * Sx, 4)
        pred_reg_flatten, pred_cls_flatten, _ = output_flattening(pred_reg, pred_cls)
        targ_reg_flatten, targ_cls_flatten, _ = output_flattening(targ_reg, targ_cls)
        gt_pos_index = torch.nonzero(targ_cls_flatten == 1).flatten()  # (n_pos,)
        gt_neg_index = torch.nonzero(targ_cls_flatten == -1).flatten() # (n_neg,)
        N_pos = len(gt_pos_index)
        N_neg = len(gt_neg_index)
        
        ### cls ### ~ length min(M/2, N_pos) for pos and min(M - N_pos, N_neg) for neg
        # Extract M/2 pos prediction label (all if smaller)
        N_get_pos = np.minimum(effective_batch // 2, N_pos).item()
        choosen_pos_index =  gt_pos_index[np.random.choice(N_pos, N_get_pos, replace=False)]
        pred_cls_flatten_pos_choosen = pred_cls_flatten[choosen_pos_index]
        # Get randomly choosen negative prediction labels (all if total - N_pos > N_neg or total - N_pos < N_neg)
        N_get_neg = np.minimum(effective_batch - N_get_pos, N_neg).item()
        choosen_neg_index = gt_neg_index[np.random.choice(N_neg, N_get_neg, replace=False)]
        pred_cls_flatten_neg_choosen = pred_cls_flatten[choosen_neg_index]
        
        ### reg ### ~ length min(M, N_pos)
        # Extract pos pred regression
        N_get_pos_reg = np.minimum(effective_batch, N_pos).item()
        choosen_pos_index =  gt_pos_index[np.random.choice(N_pos, N_get_pos_reg, replace=False)] # torch.randperm(N_get_pos_reg)
        pred_reg_flatten_choosen = pred_reg_flatten[choosen_pos_index]
        # Extract pos gt regression
        targ_reg_flatten_choosen = targ_reg_flatten[choosen_pos_index]

        # Calculate classification and regression loss
        loss_c, N_cls = self.loss_class(pred_cls_flatten_pos_choosen, pred_cls_flatten_neg_choosen)
        loss_r, N_reg = self.loss_reg(targ_reg_flatten_choosen, pred_reg_flatten_choosen)
        # Adjust weight factor to get equally weighted cls and reg loss
        l = N_reg / N_cls
        total_loss = loss_c + l * loss_r
        return total_loss, loss_c, loss_r


    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        '''
        Post process for the outputs for a batch of images
        Input:
            out_c:  (bz,1,grid_size[0],grid_size[1])
            out_r:  (bz,4,grid_size[0],grid_size[1])
            IOU_thresh: scalar that is the IOU threshold for the NMS
            keep_num_preNMS: number of masks we will keep from each image before the NMS
            keep_num_postNMS: number of masks we will keep from each image after the NMS
        Output:
            nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
            nms_box_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
        '''
        # Iterate through each image in the batch and apply thresholding and NMS
        assert len(out_c.shape) == 4

        nms_cls_list, nms_box_list, pre_nms_cls_list, pre_nms_coord_list, top_20_cls_list, top_20_coord_list = [], [], [], [], [], []
        for batch_idx in range(len(out_c)):
            nms_cls, nms_box, pre_nms_cls, pre_nms_coord, top_20_cls, top_20_coord = self.postprocessImg(out_c[batch_idx], 
                                                                                                         out_r[batch_idx], 
                                                                                                         IOU_thresh, 
                                                                                                         keep_num_preNMS, 
                                                                                                         keep_num_postNMS)
            nms_cls_list.append(nms_cls)
            nms_box_list.append(nms_box)
            pre_nms_cls_list.append(pre_nms_cls)
            pre_nms_coord_list.append(pre_nms_coord)
            top_20_cls_list.append(top_20_cls)
            top_20_coord_list.append(top_20_coord)

        return nms_cls_list, nms_box_list, pre_nms_cls_list, pre_nms_coord_list, top_20_cls_list, top_20_coord_list


    def postprocessImg(self, pred_cls, pred_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        '''
        Post process the output for one image
        Input:
            pred_cls: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
            pred_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
        Output:
            nms_clas: (Post_NMS_boxes)
            nms_box: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
            bound_top_K_cls: (K,) Classification score before NMS
            bound_top_K_coord: (k,4) Bbox coordinate prediction before NMS
            top_20_cls: (20,) Top 20 proposals score before NMS
            top_20_coord: (20,4) Top 20 proposals box coordinate before NMS
        '''
        # Decode the predicted bboxes
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(pred_coord.unsqueeze(0), pred_cls.unsqueeze(0), self.get_anchors())
        decoded_coord = output_decoding(flatten_coord, flatten_anchors) # (N,4)
        # Clip the crossing boundary boxes
        bbox_x_min = decoded_coord[:,0]
        bbox_y_min = decoded_coord[:,1]
        bbox_x_max = decoded_coord[:,2]
        bbox_y_max = decoded_coord[:,3]
        x_valid = torch.logical_and(bbox_x_min >=0, bbox_x_max < 1088)
        y_valid = torch.logical_and(bbox_y_min >=0, bbox_y_max < 800)
        boundary_valid_mask = torch.logical_and(x_valid, y_valid) # (N,)
        # Filter out cross boundary bbox
        bound_valid_coord = decoded_coord[boundary_valid_mask]
        bound_valid_cls = flatten_gt[boundary_valid_mask]
        # Sort the classification score and choose top K anchors
        sorted_score_index = torch.argsort(bound_valid_cls, descending=True)
        top_K_index = sorted_score_index[:keep_num_preNMS]
        bound_top_K_coord = bound_valid_coord[top_K_index] # (K, 4)
        bound_top_K_cls = bound_valid_cls[top_K_index]     # (K,)
        # Also return the top 20 porposals 
        top_20_cls = bound_valid_cls[sorted_score_index[:20]]
        top_20_coord = bound_valid_coord[sorted_score_index[:20]]
        # Perform NMS to further eliminate highly overlaped boxes
        nms_cls, nms_box = self.NMS(bound_top_K_cls, bound_top_K_coord, IOU_thresh)
        # Select the top N boxes after NMS
        nms_cls = nms_cls[:keep_num_postNMS]
        nms_box = nms_box[:keep_num_postNMS]
        return nms_cls, nms_box, bound_top_K_cls, bound_top_K_coord, top_20_cls, top_20_coord


    def NMS(self, clas, prebox, thresh):
        '''
        Input:
            clas: (top_k_boxes) (scores of the top k boxes)
            prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        Output:
            nms_cls: (Post_NMS_boxes)
            nms_posbox: (Post_NMS_boxes,4)
        '''
        # Convert xyxy bbox to cxcywh and calculate cross-IOU
        cxcywh_box = ops.box_convert(prebox, 'xyxy', 'cxcywh')
        cr_IOU = calc_IOU(cxcywh_box, cxcywh_box).tril(-1)
        # Iterate through all bboxes and keep record of suppressed bbox's index
        N = len(cxcywh_box)
        discard_index = []
        for i in range(N):
            discard_index.extend(torch.nonzero(cr_IOU[:,i] > thresh).flatten().tolist())
        all_index_set = set(range(N))
        remained_index = list(all_index_set.difference(discard_index))
        # Extract non-suppressed bbox and score
        nms_cls = clas[remained_index]
        nms_box = prebox[remained_index]
        return nms_cls, nms_box


    def visualNMS(self, rpn_net, images, boxes):
        '''
        Visualize the proposed bounding box with gt bounding box before and after NMS
        images: (3,800,1088)
        '''
        cls_pred, reg_pred = rpn_net.forward(images.unsqueeze(0).cuda()) # (1,1,50,68), (1,4,50,68)
        # plt.imshow(cls_pred.squeeze().cpu().detach().numpy())
        nms_cls_list, nms_box_list, pre_nms_cls_list, pre_nms_coord_list, top_20_cls_list, top_20_coord_list = rpn_net.postprocess(cls_pred.cpu().detach(), reg_pred.cpu().detach())
        nms_cls = nms_cls_list[0]
        nms_box = nms_box_list[0]
        pre_nms_cls = pre_nms_cls_list[0]
        pre_nms_coord = pre_nms_coord_list[0]
        top_20_cls = top_20_cls_list[0]
        top_20_coord = top_20_coord_list[0]
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
        img = untransform(images.clone()).numpy().astype(np.float64).transpose(1,2,0)
        img[:,:11,:] = 0
        img[:,-11:,:] = 0
        fig, ax = plt.subplots(1,2, figsize=(15,8))
        # Plot the image along with ground truth bbox and bbox before NMS
        ax[0].imshow(img)
        ax[0].set_title('Before NMS')
        for idx in range(len(pre_nms_cls)):
            coord = pre_nms_coord[idx]
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='b')
            ax[0].add_patch(rect)
        for gt_idx in range(len(boxes)):
            coord = boxes[gt_idx]
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r')
            ax[0].add_patch(rect)
        # Plot the image along with ground truth bbox and bbox after NMS
        ax[1].imshow(img)
        ax[1].set_title('Ater NMS')
        for coord in nms_box:
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='b')
            ax[1].add_patch(rect)
        for gt_idx in range(len(boxes)):
            coord = boxes[gt_idx]
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r')
            ax[1].add_patch(rect)


    def visualTop20(self, rpn_net, images, boxes):
        '''
        Visualize the top 20 proposed bounding boxes before NMS
        '''
        cls_pred, reg_pred = rpn_net.forward(images.unsqueeze(0).cuda()) # (1,1,50,68), (1,4,50,68)
        nms_cls_list, nms_box_list, pre_nms_cls_list, pre_nms_coord_list, top_20_cls_list, top_20_coord_list = rpn_net.postprocess(cls_pred.cpu().detach(), reg_pred.cpu().detach())
        nms_cls = nms_cls_list[0]
        nms_box = nms_box_list[0]
        pre_nms_cls = pre_nms_cls_list[0]
        pre_nms_coord = pre_nms_coord_list[0]
        top_20_cls = top_20_cls_list[0]
        top_20_coord = top_20_coord_list[0]
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
        img = untransform(images.clone()).numpy().astype(np.float64).transpose(1,2,0)
        img[:,:11,:] = 0
        img[:,-11:,:] = 0
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        # Plot the image along with ground truth bbox and bbox before NMS
        ax.imshow(img)
        ax.set_title('Top 20 Proposals')
        for idx in range(len(top_20_cls)):
            coord = top_20_coord[idx]
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='b')
            ax.add_patch(rect)
        for gt_idx in range(len(boxes)):
            coord = boxes[gt_idx]
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color='r')
            ax.add_patch(rect)

    
    #####################
    # Pytorch-Lightning #
    #####################
    def training_step(self, batch, batch_idx):
        images = batch['images']
        indexes = batch['index']
        boxes = batch['bbox']
        # Forward of RPN model
        pred_class, pred_coord = self.forward(images)
        # Ground truth of current batch
        gt_class, gt_coord = self.create_batch_truth(boxes, indexes, images.shape[-2:])
        # Calculate the loss
        total_loss, loss_c, loss_r = self.compute_loss(pred_class, pred_coord, gt_class, gt_coord)
        # Append loss
        self.total_loss.append(total_loss.item())
        self.cls_loss.append(loss_c.item())
        self.reg_loss.append(loss_r.item())
        return total_loss


    def validation_step(self, batch, batch_idx):
        images = batch['images']
        indexes = batch['index']
        boxes = batch['bbox']
        # Forward of RPN model
        pred_class, pred_coord = self.forward(images)
        # Ground truth of current batch
        gt_class, gt_coord = self.create_batch_truth(boxes, indexes, images.shape[-2:])
        # Calculate the loss 
        total_loss, _, _ = self.compute_loss(pred_class, pred_coord, gt_class, gt_coord)
        # Append loss
        self.val_loss.append(total_loss.item())
        self.log("val_loss", total_loss)
    

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        return optimizer
    
# if __name__=="__main__":
