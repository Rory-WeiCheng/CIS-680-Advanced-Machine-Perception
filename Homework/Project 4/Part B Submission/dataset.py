import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *


class BuildDataset(Dataset):
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
                 masks:  list:len(bz) {(n_obj, 800,1088)}
                 bbox:   list:len(bz) {(n_obj, 4)
                 index:  list:len(bz)
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