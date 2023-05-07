import numpy as np
import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

np.random.seed(17)


class SOLO_Dataset(Dataset):
    def __init__(self, img_path, mask_path, bbox_path, label_path, indices=[]):
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
        self.indices = indices


    def __len__(self):
        if len(self.indices) > 0:
            return len(self.indices)
        else:
            return self.label.shape[0]


    def __getitem__(self, idx):
        '''
        Input:
        idx: int
        Output:
        img: (3, 800, 1088) Tensor
        label: (n_obj,) Tensor
        mask: (n_obj, 800, 1088) Tensor
        bbox: (n_obj, 4) Tensor
        '''
        if len(self.indices) > 0:
            idx = self.indices[idx]
        # Process image
        img_ith = self.img[idx].astype(np.uint8).transpose(1,2,0) # from (C,H,W) to (H,W,3)
        img = self.img_transform(img_ith)
        # Process label
        label = torch.from_numpy(self.label[idx])
        # Process bounding box
        bbox = torch.from_numpy(self.bbox[idx] * self.bbox_transform_ratio)
        # Process mask
        acc_num = 0
        for i in range(idx):
            curr_label = self.label[i]
            acc_num += len(curr_label)
        num_obj = len(self.label[idx])
        mask_ith = torch.from_numpy(self.mask[acc_num : acc_num + num_obj].astype(np.uint8))
        mask = self.mask_transform(mask_ith)

        return img, label, mask, bbox



class HW3DataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size=2, val_batch_size=2):
        super().__init__()
        self.data_len = 3265
        self.data_train = None
        self.data_val = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.indices = np.random.permutation(self.data_len)
    
    def train_dataloader(self):
        if self.data_train is None:
            self.data_train = SOLO_Dataset(
                img_path='data/hw3_mycocodata_img_comp_zlib.h5',
                mask_path='data/hw3_mycocodata_mask_comp_zlib.h5',
                bbox_path='data/hw3_mycocodata_bboxes_comp_zlib.npy',
                label_path='data/hw3_mycocodata_labels_comp_zlib.npy',
                indices=self.indices[:int(0.8*self.data_len)]
            )
        return DataLoader(self.data_train, batch_size=self.train_batch_size, num_workers=8, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        if self.data_val is None:
            self.data_val = SOLO_Dataset(
                img_path='data/hw3_mycocodata_img_comp_zlib.h5',
                mask_path='data/hw3_mycocodata_mask_comp_zlib.h5',
                bbox_path='data/hw3_mycocodata_bboxes_comp_zlib.npy',
                label_path='data/hw3_mycocodata_labels_comp_zlib.npy',
                indices=self.indices[int(0.8*self.data_len):]
            )
        return DataLoader(self.data_val, batch_size=self.val_batch_size, num_workers=8, collate_fn=collate_fn)
    

def collate_fn(batch):
    images, labels, masks, bounding_boxes = list(zip(*batch))
    return torch.stack(images), labels, masks, bounding_boxes


##############################################
# Data loading of previous version (not used)#
##############################################

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