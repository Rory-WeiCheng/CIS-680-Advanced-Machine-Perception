import torchvision
import torch
import numpy as np
from BoxHead import *
from utils import *
from pretrained_models import *
from torchvision.models.detection.image_list import ImageList


if __name__ == '__main__':
    # 1. Load pretrained backbone and RPN network
    pretrained_path='data/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path, device=device)
    # 2. Load given hold_out_images.npz file for test
    hold_images_path = 'data/hold_out_images.npz'
    test_images = np.load(hold_images_path, allow_pickle=True)['input_images']
    # 3. Load boxHead model 
    boxHead = BoxHead(device=device)
    boxHead = boxHead.to(device)
    boxHead.eval()
    # Get pre-trained weight
    train_model_path = 'train_epoch39'
    checkpoint = torch.load(train_model_path)
    boxHead.load_state_dict(checkpoint['box_head_state_dict'])
    keep_topK = 200
    # 4. Iterate through all images in given file to perform classification
    cpu_boxes = []
    cpu_scores = []
    cpu_labels = []
    for i, numpy_image in enumerate(test_images, 0):
        images = torch.from_numpy(numpy_image).to(device)
        with torch.no_grad():
            # 4.1: Extract the feature pyramid from backbone FPN
            backout = backbone(images)
            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
            
            # 4.2: Pass the image list and get proposed region from backbone RPN
            rpnout = rpn(im_lis, backout)
            # A list of proposal tensors: list:len(bz) {(keep_topK,4)}
            proposals = [proposal[0:keep_topK,:] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN) {(bz,256,H_feat,W_feat)}
            fpn_feat_list = list(backout.values())

            # 4.3: Perform ROI Aign based on FPN ouptut and RPN proposal
            feature_vectors = boxHead.MultiScaleRoiAlign(fpn_feat_list, proposals)

            # 4.4: boxHead forward for classification
            class_logits, box_pred = boxHead(feature_vectors)

            # 4.5: Post-processing based on boxHead model output
            boxes, scores, labels = boxHead.postprocess_detections(class_logits,
                                                                   box_pred,
                                                                   proposals,
                                                                   conf_thresh=0.8, 
                                                                   keep_num_preNMS=200, 
                                                                   keep_num_postNMS=3)
            # 4.6: Iterate and save all results
            for box, score, label in zip(boxes, scores, labels):
                if box is None:
                    cpu_boxes.append(None)
                    cpu_scores.append(None)
                    cpu_labels.append(None)
                else:
                    cpu_boxes.append(box.to('cpu').detach().numpy())
                    cpu_scores.append(score.to('cpu').detach().numpy())
                    cpu_labels.append(label.to('cpu').detach().numpy())
    np.savez('predictions.npz', predictions={'boxes': cpu_boxes, 'scores': cpu_scores,'labels': cpu_labels})
