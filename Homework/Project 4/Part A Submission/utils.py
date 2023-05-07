import numpy as np
import torch
from functools import partial

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))


def calc_IOU(prop_mat, gt_mat):
    '''
    Calculate the IOU between each bbox in prop_mat with gt-bbox in vectorization
    Input:
        prop_mat: (N,4) [cx, cy, w, h] in pixel
        gt_mat:   (M,4) [cx, cy, w, h] in pixel
    Return:
        IoU: (N,M)
    '''
    # Prediction
    x_p_min = prop_mat[:,0:1] - prop_mat[:,2:3] / 2 # (N,1)
    x_p_max = prop_mat[:,0:1] + prop_mat[:,2:3] / 2 # (N,1)
    y_p_min = prop_mat[:,1:2] - prop_mat[:,3:4] / 2 # (N,1)
    y_p_max = prop_mat[:,1:2] + prop_mat[:,3:4] / 2 # (N,1)
    # Ground truth
    x_gt_min = gt_mat[:,0:1] - gt_mat[:,2:3] / 2 # (M,1)
    x_gt_max = gt_mat[:,0:1] + gt_mat[:,2:3] / 2 # (M,1)
    y_gt_min = gt_mat[:,1:2] - gt_mat[:,3:4] / 2 # (M,1)
    y_gt_max = gt_mat[:,1:2] + gt_mat[:,3:4] / 2 # (M,1)
    # Compute the anchor area for prediction
    prop_mat_w = prop_mat[:,2] # (N,)
    prop_mat_h = prop_mat[:,3] # (N,)
    area_p = (prop_mat_w * prop_mat_h).reshape(-1,1) # (N,1)
    # Compute the anchor area for ground truth
    gt_mat_w = gt_mat[:,2] # (M,)
    gt_mat_h = gt_mat[:,3] # (M,)
    area_gt = (gt_mat_w * gt_mat_h).reshape(-1,1) # (M,1)
    # Compute the intersect w & h
    zero_arr = torch.zeros(prop_mat.shape[0], gt_mat.shape[0]).to(prop_mat.device)
    intersect_p_gt_w = torch.maximum(torch.minimum(x_p_max, x_gt_max.T) - torch.maximum(x_p_min, x_gt_min.T), zero_arr) # (N,4)
    intersect_p_gt_h = torch.maximum(torch.minimum(y_p_max, y_gt_max.T) - torch.maximum(y_p_min, y_gt_min.T), zero_arr) # (N,4)
    # compute intersect area
    area_intersect = intersect_p_gt_w * intersect_p_gt_h # (N,4)
    # compute union area
    area_union = (area_p + area_gt.T) - area_intersect #(N,4)
    # Compute final IoU
    IoU = area_intersect / area_union #(N,M)

    return IoU


def output_flattening(ground_coord, gt, anchors=None):
    '''
    This function flattens the output of the network and the corresponding anchors 
    in the sense that it concatenates the outputs and the anchors from all the grid cells
    from all the images into 2D matrices
    Each row of the 2D matrices corresponds to a specific anchor/grid cell
    Input:
        ground_coord:   (bz, 4, grid_size[0], grid_size[1])
        gt:             (bz, 1, grid_size[0], grid_size[1])
        anchors:        (grid_size[0], grid_size[1], 4)
    Output:
        flatten_regr:    (bz * grid_size[0] * grid_size[1], 4)
        flatten_clas:    (bz * grid_size[0] * grid_size[1],)
        flatten_anchors: (bz * grid_size[0] * grid_size[1], 4)
    '''
    bz = ground_coord.shape[0]
    flatten_regr = ground_coord.flatten(2).permute(0,2,1).reshape(-1,4)
    flatten_clas = gt.flatten(2).permute(0,2,1).flatten()
    if anchors != None:
        flatten_anchors = anchors.view(-1,4).repeat(bz,1)
    else:
        flatten_anchors = -1
    return flatten_regr, flatten_clas, flatten_anchors


def output_decoding(flatten_coord, flatten_anchors, device='cpu'):
    '''
    This function decodes the output that is given in the encoded format (defined in the handout)
    into box coordinates where it returns the upper left and lower right corner of the proposed box
    Input:
        flatten_coord:      (total_number_of_anchors * bz, 4)
        flatten_anchors:    (total_number_of_anchors * bz, 4)
    Output:
        box: (total_number_of_anchors * bz, 4)
    '''
    cxcywh_box = torch.zeros(flatten_coord.shape[0], 4)
    box = torch.zeros(flatten_coord.shape[0], 4)
    # Decoding the cx, cy, w, h information
    cxcywh_box[:, 0] = flatten_coord[:,0] * flatten_anchors[:,2] + flatten_anchors[:,0]
    cxcywh_box[:, 1] = flatten_coord[:,1] * flatten_anchors[:,3] + flatten_anchors[:,1]
    cxcywh_box[:, 2] = torch.exp(flatten_coord[:,2]) * flatten_anchors[:,2]
    cxcywh_box[:, 3] = torch.exp(flatten_coord[:,3]) * flatten_anchors[:,3]
    # Get the xyxy coordinates from cxcywh format
    box[:, 0] = cxcywh_box[:,0] - cxcywh_box[:,2] / 2
    box[:, 1] = cxcywh_box[:,1] - cxcywh_box[:,3] / 2
    box[:, 2] = cxcywh_box[:,0] + cxcywh_box[:,2] / 2
    box[:, 3] = cxcywh_box[:,1] + cxcywh_box[:,3] / 2
    return box