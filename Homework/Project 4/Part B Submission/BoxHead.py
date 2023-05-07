import torch
from torch import nn
from utils import *
import torch.nn.functional as F
from torchvision import transforms, ops
from pretrained_models import pretrained_models_680
from torchvision.models.detection.image_list import ImageList

class BoxHead(torch.nn.Module):
    def __init__(self, Classes=3, P=7):
        super(BoxHead,self).__init__()
        self.C = Classes
        self.P = P
        
        ## TODO initialize BoxHead ##
        self.inter = torch.nn.Sequential(
            nn.Linear(in_features=256*self.P**2, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        )
        # Classifier
        self.cls = torch.nn.Sequential(
            nn.Linear(in_features=1024, out_features=self.C+1),
            # nn.Softmax()
        )
        # Regressor
        self.reg = torch.nn.Sequential(
            nn.Linear(in_features=1024, out_features=self.C*4)
        )


    def forward(self, feature_vectors, eval=False):
        '''
        Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
        Input:
            feature_vectors: (total_proposals, 256*P*P)
        Outputs:
            class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
                                                   CrossEntropyLoss you should not pass the output through softmax here)
            box_pred:     (total_proposals,4*C)
        '''
        # Intermediate layer
        inter = self.inter(feature_vectors)
        # Classifier
        class_logits = self.cls(inter)
        if eval:
            class_logits = F.softmax(class_logits, dim=1)
        # Regressor
        box_pred = self.reg(inter)
        return class_logits, box_pred


    def create_ground_truth(self, proposals, gt_labels, bboxes):
        '''
        This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
        Input:
            proposals: list:len(bz) {(per_image_proposals,4)} in [x1,y1,x2,y2] format
            gt_labels: list:len(bz) {(n_obj,)}
            bboxes:    list:len(bz) {(n_obj, 4)}
        Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals,1) (the class that the proposal is assigned)
            regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
        '''
        total_batch_label, total_reg_target = [], []
        # Iterate through each image in the batch to assign gt-label and gt-reg
        for batch_idx in range(len(proposals)):
            # proposal: (per_image_proposals,4)  gt_label: (n_obj,)  bbox: (n_obj, 4)
            proposal = proposals[batch_idx]
            gt_label = gt_labels[batch_idx].to(proposal.device)
            bbox = bboxes[batch_idx].to(proposal.device)
            # Initialize empty label and regressor container for current image
            label_ith = torch.zeros(proposal.shape[0], dtype=torch.long).to(proposal.device)
            reg_ith = torch.zeros((proposal.shape[0], 4)).to(proposal.device)
            # Calculate cross IOU between all proposed bbox and gt-bbox
            cross_IOU = IOU_xyxy(proposal, bbox.cuda())
            # Find bkgd and non-bkgd mask for all proposals
            bkgd_mask = (cross_IOU <= 0.5).all(dim=1)
            non_bkgd_mask = ~bkgd_mask
            assigned_gt_index = torch.argmax(cross_IOU, dim=1)[non_bkgd_mask]
            # Assign cls
            label_ith[non_bkgd_mask] = gt_label[assigned_gt_index]
            label_ith = label_ith.view(-1,1)
            # Assign reg
            corr_gt_box = bbox[assigned_gt_index]
            non_bkgd_prop = proposal[non_bkgd_mask]
            non_bkgd_prop_w = non_bkgd_prop[:,2] - non_bkgd_prop[:,0]
            non_bkgd_prop_h = non_bkgd_prop[:,3] - non_bkgd_prop[:,1]
            reg_ith[non_bkgd_mask, 0] = (corr_gt_box[:,0] - non_bkgd_prop[:,0]) / non_bkgd_prop_w
            reg_ith[non_bkgd_mask, 1] = (corr_gt_box[:,1] - non_bkgd_prop[:,1]) / non_bkgd_prop_h
            reg_ith[non_bkgd_mask, 2] = torch.log((corr_gt_box[:,2] - corr_gt_box[:,0]) / non_bkgd_prop_w)
            reg_ith[non_bkgd_mask, 3] = torch.log((corr_gt_box[:,3] - corr_gt_box[:,1]) / non_bkgd_prop_h)
            # Append current image result to total batch
            total_batch_label.append(label_ith)
            total_reg_target.append(reg_ith)
        # Concatenate all images in the batch
        labels = torch.concat(total_batch_label)
        regressor_target = torch.concat(total_reg_target)
        return labels, regressor_target


    def MultiScaleRoiAlign(self, fpn_feat_list, proposals, P=7):
        '''
        This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        Input:
            fpn_feat_list: list:len(FPN) {(bz, 256, H_feat, W_feat)}
            proposals:     list:len(bz) {(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            P:             scalar
        Output:
            feature_vectors: (total_proposals, 256*P*P) (make sure the ordering of the proposals are the same as 
                                                        the ground truth creation)
        '''
        feature_vectors = []
        # Iterate through each image in the batch
        for batch_idx in range(len(proposals)):
            proposal = proposals[batch_idx] # (per_image_proposals, 4)
            curr_feature_vector = torch.zeros(proposal.shape[0], 256*P*P).to(proposal.device)
            w = proposal[:,2] - proposal[:,0]
            h = proposal[:,3] - proposal[:,1]
            k = torch.clip(torch.floor(4 + torch.log2(torch.sqrt(w * h)/224)), min=2, max=5).to(torch.long) - 2
            for level_index in range(4):
                curr_level_mask = torch.nonzero(k == level_index).flatten()
                if len(curr_level_mask) > 0:
                    corr_feature_map = fpn_feat_list[level_index][batch_idx:batch_idx+1] # (1, 256, H_feat, W_feat)
                    prop_bbox = proposal[curr_level_mask]
                    scale_ratio = corr_feature_map.shape[-1] / 1088
                    prop_bbox_fm = prop_bbox * scale_ratio
                    # Perform ROI Align and append result
                    roi_align_res = ops.roi_align(corr_feature_map, [prop_bbox_fm], [P,P], sampling_ratio=2).flatten(1)
                    curr_feature_vector[curr_level_mask] = roi_align_res
            feature_vectors.append(curr_feature_vector)
        # Stack all flattened proposed ROI Align results
        feature_vectors = torch.concat(feature_vectors)
        return feature_vectors


    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=100, keep_num_postNMS=50,apply_NMS=True):
        '''
        This function does the post processing for the results of the Box Head for a batch of images
        Use the proposals to distinguish the outputs from each image
        Input:
            class_logits:       (total_proposals,(C+1))
            box_regression:     (total_proposal,4*C) in [t_x,t_y,t_w,t_h] format
            proposals:          list:len(bz) (per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
            conf_thresh:        scalar
            keep_num_preNMS:    scalar (number of boxes to keep pre NMS)
            keep_num_postNMS:   scalar (number of boxes to keep post NMS)
        Output:
            pred_bboxes:  list:len(bz) {(post_NMS_boxes_per_image, 4)} in [x1,y1,x2,y2] format
            pred_scores: list:len(bz) {(post_NMS_boxes_per_image,)}   (the score for the top class for the regressed box)
            pred_labels: list:len(bz) {(post_NMS_boxes_per_image,)}   (top class of each regressed box)
        '''
        # Iterate through each batch
        start_index = 0
        pred_bboxes, pred_scores, pred_labels = [], [], []
        for batch_idx in range(len(proposals)):
            # Extract current batch's classification and regression
            N_prop = len(proposals[batch_idx])
            batch_cls = class_logits[start_index:start_index+N_prop]
            batch_reg = box_regression[start_index:start_index+N_prop]
            batch_prop = proposals[batch_idx]
            start_index += N_prop
            # Remove proposals with background confidence larger than 0.5 or in most confidence
            fgd_conf_mask = batch_cls[:,0] < conf_thresh # (per_image_proposals,)
            fgd_most_mask = torch.argmax(batch_cls, dim=1) != 0
            total_mask = torch.logical_and(fgd_conf_mask, fgd_most_mask)
            batch_cls = batch_cls[total_mask]
            batch_reg = batch_reg[total_mask]
            batch_prop = batch_prop[total_mask]
            # Decode bounding box
            pred_conf, pred_label = torch.max(batch_cls, dim=1)
            take_row_index = torch.Tensor(list(range(batch_reg.shape[0]))).to(torch.long).repeat(4,1).T
            take_col_index = (pred_label - 1).repeat(4,1).T * 4
            for i in range(4):
                take_col_index[:,i] += i
            pred_coord = batch_reg[take_row_index, take_col_index]
            decoded_coord = output_decoding(pred_coord, ops.box_convert(batch_prop, 'xyxy', 'cxcywh'))
            # Remove cross-boundary box
            bbox_x_min = decoded_coord[:,0]
            bbox_y_min = decoded_coord[:,1]
            bbox_x_max = decoded_coord[:,2]
            bbox_y_max = decoded_coord[:,3]
            x_valid = torch.logical_and(bbox_x_min >=0, bbox_x_max < 1088)
            y_valid = torch.logical_and(bbox_y_min >=0, bbox_y_max < 800)
            boundary_valid_mask = torch.logical_and(x_valid, y_valid)
            # Final masking
            pred_coord = decoded_coord[boundary_valid_mask]
            pred_label = pred_label[boundary_valid_mask]
            pred_conf = pred_conf[boundary_valid_mask]
            batch_prop = batch_prop[boundary_valid_mask]
            # Sort and keep top K boxes before NMS
            sorted_score_index = torch.argsort(pred_conf, descending=True)
            top_K_index = sorted_score_index[:keep_num_preNMS]
            bound_top_K_coord = pred_coord[top_K_index] # (K, 4)
            bound_top_K_cls = pred_conf[top_K_index]    # (K,)
            bound_top_K_label = pred_label[top_K_index] # (K,)
            if apply_NMS:
                # Apply NMS 
                nms_cls, nms_box, nms_label = self.NMS(bound_top_K_cls, bound_top_K_coord, bound_top_K_label)
                # Append current batch result
                pred_bboxes.append(nms_box[:keep_num_postNMS])
                pred_scores.append(nms_cls[:keep_num_postNMS])
                pred_labels.append(nms_label[:keep_num_postNMS])
            else:
                pred_bboxes.append(bound_top_K_coord)
                pred_scores.append(bound_top_K_cls[:keep_num_postNMS])
                pred_labels.append(bound_top_K_label[:keep_num_postNMS])
        return pred_bboxes, pred_scores, pred_labels


    def NMS(self, clas, prebox, label, thresh=0.5):
        '''
        Input:
            clas: (top_k_boxes) (scores of the top k boxes)
            prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        Output:
            nms_cls: (Post_NMS_boxes)
            nms_posbox: (Post_NMS_boxes,4)
        '''
        # Calculate cross-IOU
        cr_IOU = IOU_xyxy(prebox, prebox).tril(-1)
        # Iterate through all bboxes and keep record of suppressed bbox's index
        N = len(prebox)
        discard_index = []
        for i in range(N):
            discard_index.extend(torch.nonzero(cr_IOU[:,i] > thresh).flatten().tolist())
        all_index_set = set(range(N))
        remained_index = list(all_index_set.difference(discard_index))
        # Extract non-suppressed bbox and score
        nms_cls = clas[remained_index]
        nms_box = prebox[remained_index]
        nms_label = label[remained_index]
        return nms_cls, nms_box, nms_label


    def compute_loss(self, pred_cls, pred_reg, gt_cls, gt_reg, l=1, effective_batch=40):
        '''
        Compute the total loss of the classifier and the regressor
        Input:
            pred_cls:        (total_proposals, (C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
            pred_reg:        (total_proposals, 4*C)      (as outputed from forward)
            gt_cls:          (total_proposals, 1)
            gt_reg:          (total_proposals, 4)
            l:               scalar (weighting of the two losses)
            effective_batch: scalar
        Output:
            loss:       scalar
            loss_class: scalar
            loss_regr:  scalar
        '''
        # Sample non-background and background mini-batches with 3:1 ratio
        gt_fgd_index = torch.nonzero(gt_cls.flatten() != 0).flatten() # (n_fgd,)
        gt_bgd_index = torch.nonzero(gt_cls.flatten() == 0).flatten() # (n_bgd,)
        N_fgd = len(gt_fgd_index)
        N_bgd = len(gt_bgd_index)
        
        # Get randomly selected fgd index
        N_get_fgd = np.minimum(effective_batch*3//4, N_fgd).item()
        choosen_fgd_index = gt_fgd_index[np.random.choice(N_fgd, N_get_fgd, replace=False)]
        # Get randomly selected bgd index
        N_get_bgd = np.minimum(effective_batch - N_get_fgd, N_bgd).item()
        choosen_bgd_index = gt_bgd_index[np.random.choice(N_bgd, N_get_bgd, replace=False)]
        ## Classifier's CrossEntropyLoss ##
        CELoss = torch.nn.CrossEntropyLoss(reduction='sum')
        fgd_pred_cls = pred_cls[choosen_fgd_index]
        fgd_gt_cls = gt_cls[choosen_fgd_index].flatten()
        bgd_pred_cls = pred_cls[choosen_bgd_index]
        bgd_gt_cls = gt_cls[choosen_bgd_index].flatten()
        loss_cls = CELoss(fgd_pred_cls, fgd_gt_cls) + CELoss(bgd_pred_cls, bgd_gt_cls)
        N_cls = len(fgd_pred_cls) + len(bgd_pred_cls)
        loss_cls = loss_cls / N_cls
        
        # Get randomly selected index from fgd_index to compute regressor loss
        N_get_fgd_reg = np.minimum(effective_batch, N_fgd).item()
        choosen_fgd_index = gt_fgd_index[np.random.choice(N_fgd, N_get_fgd_reg, replace=False)]
        fgd_pred_reg = pred_reg[choosen_fgd_index] # (M, 4*C)
        fgd_gt_reg = gt_reg[choosen_fgd_index] #(M, 4)
        fgd_pred_reg_cls = gt_cls[choosen_fgd_index].flatten() # (M,)
        ## Regressor's loss ##
        SmoothL1 = torch.nn.SmoothL1Loss()
        N_pred = fgd_pred_reg.shape[0]
        take_row_index = torch.Tensor(list(range(N_pred))).to(torch.long).repeat(4,1).T # (M,4)
        take_col_index = (fgd_pred_reg_cls - 1).repeat(4,1).T * 4 # (M,4)
        for i in range(4):
            take_col_index[:,i] += i
        taken_fgd_pred_reg = fgd_pred_reg[take_row_index, take_col_index] # (M, 4)
        N_reg = len(taken_fgd_pred_reg)
        loss_reg = SmoothL1(taken_fgd_pred_reg, fgd_gt_reg)

        ## Total loss ##
        l = N_reg / N_cls
        loss = loss_cls + l * loss_reg

        return loss, loss_cls, loss_reg


'''
    # # #####################
    # # # Pytorch-Lightning #
    # # #####################
    # def training_step(self, batch, batch_idx):
    #     self.backbone.eval()
    #     self.rpn.eval()
    #     # 1. Extract current batch
    #     images = batch['images'].cuda()
    #     labels = batch['labels']
    #     boxes = batch['bbox']
    #     # 2. Get proposals and feature pyramid from pretrained FPN and RPN
    #     backout = self.backbone(images)
    #     im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    #     rpnout = self.rpn(im_lis, backout)
    #     keep_topK = 200
    #     proposals = [proposal[0:keep_topK,:] for proposal in rpnout[0]] # list:len(bz) {(keep_topK,4)}
    #     fpn_feat_list = list(backout.values()) # list:len(FPN) {(bz,256,H_feat,W_feat)}
    #     # 3. Forward of boxHead model
    #     feature_vectors = self.MultiScaleRoiAlign(fpn_feat_list, proposals)
    #     pred_class, pred_coord = self.forward(feature_vectors)
    #     # 4. Get corresponding GT cls and reg to compute loss
    #     gt_labels, gt_reg = self.create_ground_truth(proposals, labels, boxes)
    #     loss, loss_cls, loss_reg = self.compute_loss(pred_class, pred_coord, gt_labels, gt_reg)
    #     # 5. Keep record of total loss, cls loss and reg loss
    #     self.total_loss.append(loss.item())
    #     self.cls_loss.append(loss_cls.item())
    #     self.reg_loss.append(loss_reg.item())
    #     return loss


    # def validation_step(self, batch, batch_idx):
    #     # 1. Extract current batch
    #     images = batch['images'].cuda()
    #     labels = batch['labels']
    #     boxes = batch['bbox']
    #     # 2. Get proposals and feature pyramid from pretrained FPN and RPN
    #     backout = self.backbone(images)
    #     im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    #     rpnout = self.rpn(im_lis, backout)
    #     keep_topK = 200
    #     proposals = [proposal[0:keep_topK,:] for proposal in rpnout[0]] # list:len(bz) {(keep_topK,4)}
    #     fpn_feat_list = list(backout.values()) # list:len(FPN) {(bz,256,H_feat,W_feat)}
    #     # 3. Forward of boxHead model
    #     feature_vectors = self.MultiScaleRoiAlign(fpn_feat_list, proposals)
    #     pred_class, pred_coord = self.forward(feature_vectors)
    #     # 4. Get corresponding GT cls and reg to compute loss
    #     gt_labels, gt_reg = self.create_ground_truth(proposals, labels, boxes)
    #     val_loss, _, _ = self.compute_loss(pred_class, pred_coord, gt_labels, gt_reg)
    #     # 5. Keep record of validation loss
    #     self.val_loss.append(val_loss.item())
    #     self.log("val_loss", val_loss)

    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    #     return optimizer
'''

