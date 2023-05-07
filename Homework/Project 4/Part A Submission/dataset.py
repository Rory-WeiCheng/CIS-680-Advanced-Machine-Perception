import h5py
import cv2
import numpy as np
from utils import *
from rpn import *
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        img_path, mask_path, label_path, bbox_path = paths
        ## Image ##
        f = h5py.File(img_path, 'r')
        self.img = f['data'][()]
        f.close()
        ## Mask ##
        f = h5py.File(mask_path, 'r')
        self.mask = f['data'][()]
        f.close()
        ## Bounding box ##
        self.bbox = np.load(bbox_path, allow_pickle=True)
        ## Label ##
        self.label = np.load(label_path,allow_pickle=True)
        # Transform and scaling constant
        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((800, 1066)),
                                                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
                                                 transforms.Pad(padding=(11,0), fill=0)])
        self.mask_transform = transforms.Compose([transforms.Resize((800, 1066)),
                                                  transforms.Pad(padding=(11,0), fill=0)])
        self.bbox_transform_ratio = 8 / 3


    def __len__(self):
        return len(self.img)


    def __getitem__(self, index):
        # Find specified image
        sample_img = self.img[index].astype(np.uint8).transpose(1,2,0) # from (C,H,W) to (H,W,3)
        label = torch.from_numpy(self.label[index])
        sample_bbox = torch.from_numpy(self.bbox[index])
        # Find corresponding mask
        acc_num = 0
        for i in range(index):
            curr_label = self.label[i]
            acc_num += len(curr_label)
        num_obj = len(self.label[index])
        sample_mask = torch.from_numpy(self.mask[acc_num : acc_num + num_obj].astype(np.uint8))
        # Pre-process the image, mask and bbox
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(sample_img, sample_mask, sample_bbox)
        return transed_img, label, transed_mask, transed_bbox, index


    def pre_process_batch(self, sample_img, sample_mask, sample_bbox):
        # Apply transformation
        img = self.img_transform(sample_img)
        bbox = sample_bbox * self.bbox_transform_ratio
        mask = self.mask_transform(sample_mask)
        return img, mask, bbox


    def visualSampleImage(self, dataset, index):
        '''
        Visualize the original image with mask and bounding box shown
        index: list of image indexes to be shown
        '''
        untransform = transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
        color = [np.array([0,0,1]),np.array([0,1,0]),np.array([1,0,0])] # Label: 0 - Vehicle: Blue, 1 - Human: Green, 2 - Animal: Red
        # If index is not iterable, then make it as a list
        try:
            iter(index)
        except TypeError:
            index = [index]
        # Iterate through all images
        for img_idx in index: # [5,1,28,4,45]
            img_ith, label_ith, mask_ith, bbox_ith, _ = dataset[img_idx]
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
            plt.figure(figsize=(6,6))
            plt.imshow(final_img)
            for bbox_idx in range(bbox_ith.shape[0]):
                top_x, top_y = bbox_ith[bbox_idx][0], bbox_ith[bbox_idx][1]
                width, height = bbox_ith[bbox_idx][2] - bbox_ith[bbox_idx][0], bbox_ith[bbox_idx][3] - bbox_ith[bbox_idx][1]
                plt.gca().add_patch(Rectangle((top_x, top_y), width, height, linewidth=1, edgecolor='r',facecolor='none'))


    def visualGTCreation(self, images, boxes, indexes, rpn_net):
        '''
        Visualize the ground truth bbox with all positive label anchor bboxes
        '''
        try:
            iter(indexes)
        except TypeError:
            index = [indexes]
        gt, ground_coord = rpn_net.create_batch_truth(boxes.unsqueeze(0), index, images.shape[-2:])
        # Flatten the ground truth and the anchors
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(ground_coord, gt, rpn_net.get_anchors())  
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord = output_decoding(flatten_coord, flatten_anchors)
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                [1/0.229, 1/0.224, 1/0.225], inplace=False)
        images[:,:,:11] = 0
        images[:,:,-11:] = 0
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(images.permute(1,2,0))

        find_cor = (flatten_gt==1).nonzero()
        find_neg = (flatten_gt==-1).nonzero()
                    
        for elem in find_cor:
            coord = decoded_coord[elem,:].view(-1)
            anchor = flatten_anchors[elem,:].view(-1)
            col = 'r'
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect = patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)


    def histogramScaleRatio(self):
        '''
        Visualize the scale and aspect ratio histogram of all bounding boxes within dataset
        '''
        all_bbox = self.bbox
        aspect_ratio = []
        scale = []
        for i in range(len(all_bbox)):
            curr_bbox = all_bbox[i] * 8 / 3
            width, height = curr_bbox[:,2] - curr_bbox[:,0], curr_bbox[:,3] - curr_bbox[:,1] # (n_obj,)
            scale.append(np.sqrt(width * height))
            aspect_ratio.append((width / height))
        scale = np.concatenate(scale)
        aspect_ratio = np.concatenate(aspect_ratio)
        # Plot
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        ax[0].hist(scale, bins=25)
        ax[0].set_xlabel('scale range')
        ax[0].set_ylabel('Amount')
        ax[1].hist(aspect_ratio, bins=25)
        ax[1].set_xlabel('Aspect ratio range')
        ax[1].set_ylabel('Amount')


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    def collect_fn(self, batch):
        '''
        Output:
            dict{images: (bz, 3, 800, 1088)
                 labels: list:len(bz) {(n_obj,)}
                 masks: list:len(bz) {(n_obj, 800,1088)}
                 bbox: list:len(bz) {(n_obj, 4)
                 index: list:len(bz)
                 }
        '''
        images, labels, masks, bbox, index = list(zip(*batch))
        data_batch = {"images": torch.stack(images), "labels": labels, "masks": masks, "bbox": bbox, 'index':index}
        return data_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

  
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    
    for i, batch in enumerate(train_loader,0):
        images = batch['images'][0,:,:,:]
        indexes = batch['index']
        boxes = batch['bbox']
        gt, ground_coord = rpn_net.create_batch_truth(boxes, indexes, images.shape[-2:])

        # Flatten the ground truth and the anchors
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(ground_coord, gt, rpn_net.get_anchors())
        
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord = output_decoding(flatten_coord, flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)
        images[:,:11,:] = 0
        images[:,-11:,:] = 0
        fig, ax = plt.subplots(1,1)
        ax.imshow(images.permute(1,2,0))
        
        find_cor = (flatten_gt==1).nonzero()
        find_neg = (flatten_gt==-1).nonzero()
             
        for elem in find_cor:
            coord = decoded_coord[elem,:].view(-1)
            anchor = flatten_anchors[elem,:].view(-1)

            col = 'r'
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect = patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)
        plt.show()
 
        if(i + 1 == 5):
            break