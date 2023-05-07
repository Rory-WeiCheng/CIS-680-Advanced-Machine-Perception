import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader

class SOLO_Dataset(Dataset):
    def __init__(self, img_path, mask_path, bbox_path, label_path):
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
                                                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        self.mask_transform = transforms.Resize((800, 1066))
        self.bbox_transform_ratio = 8 / 3


    def __len__(self):
        return self.bbox.shape[0]


    def __getitem__(self, idx):
        # Process image
        img_ith = self.img[idx].astype(np.uint8).transpose(1,2,0) # from (C,H,W) to (H,W,3)
        img = torch.zeros((3, 800, 1088), dtype=torch.float16)
        img[:,:,:1066] = self.img_transform(img_ith)
        # Process label
        label = torch.from_numpy(self.label[idx])
        # Process bounding box
        bbox = torch.from_numpy(self.bbox[idx] * self.bbox_transform_ratio)
        # Process mask
        acc_num = 0
        for i in range(idx):
            acc_num += len(self.label[idx])
        num_obj = len(self.label[idx])
        mask_ith = torch.from_numpy(self.mask[acc_num : acc_num + num_obj].astype(np.uint8))
        mask = torch.zeros((num_obj, 800, 1088), dtype=torch.uint8)
        mask[:,:,:1066] = self.mask_transform(mask_ith)

        return img, mask, label, bbox


def collate_fn(batch):
    images, labels, masks, bounding_boxes = list(zip(*batch))
    data_batch = {"img": torch.stack(images), "bounding_boxes": bounding_boxes, "labels": labels, "masks": masks}
    return data_batch



def loadData():
    '''
    Load the raw image, mask, label and bounding box array

    Output:
    img: (3265, 3, 300, 400)
    mask: (3843, 300, 400)
    label: (3265,)
    bbox: (3265,)
    '''
    # Cutomized directory to store raw data
    prefix = 'data/' 

    ## Image ##
    f = h5py.File(prefix + 'hw3_mycocodata_img_comp_zlib.h5', 'r')
    img = f['data'][()]
    f.close()
    ## Mask ##
    f = h5py.File(prefix + 'hw3_mycocodata_mask_comp_zlib.h5', 'r')
    mask = f['data'][()]
    f.close()

    # img = torch.load('image_proc.pt') # (3265, 3, 800, 1088)
    # mask = torch.load('mask_proc.pt') # (N,)

    ## Bounding box ##
    bbox = np.load(prefix + 'hw3_mycocodata_bboxes_comp_zlib.npy', allow_pickle=True)
    ## Label ##
    label = np.load(prefix + 'hw3_mycocodata_labels_comp_zlib.npy',allow_pickle=True)
    return img, mask, bbox, label


def transform_img(img):
    '''
    Normalize, rescale and pad raw images
    Input:
    img (3265, 3, 300, 400)
    Output:
    img_new: (3265, 3, 800, 1088)
    '''
    img_copy = img.astype(np.uint8).transpose(0,2,3,1) # (N, H, W, 3)
    img_new = torch.zeros((img.shape[0], 3, 800, 1088), dtype=torch.float16)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((800, 1066)),
                                    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
                                    transforms.Pad(padding=(11,0), fill=0)])
    # Iterate through all images and apply transformation
    for i in range(img_copy.shape[0]):
        img_new[i] = transform(img_copy[i])
    return img_new


def transform_mask(mask, label):
    '''
    Rescale and pad mask
    Input:
    mask: (3843, 300, 400)
    Output:
    mask_new: (3265,)
    '''
    mask_new = []
    finished = False
    ith_image = 0
    ith_mask = 0
    transform = transforms.Compose([transforms.Resize((800, 1066)),
                                    transforms.Pad(padding=(11,0), fill=0)])
    # Iterate through all masks and group multiple masks together if they come from same image
    while ith_image != label.shape[0]:
        N_obj = len(label[ith_image])
        mask_origin = torch.from_numpy(mask[ith_mask:ith_mask+N_obj].astype(np.uint8))
        mask_curr = transform(mask_origin)
        mask_new.append(mask_curr)
        ith_image += 1
        ith_mask += N_obj
    return mask_new


def transform_bbox(bbox):
    '''
    Return the rescaled bounding box
    '''
    return bbox * 8 / 3