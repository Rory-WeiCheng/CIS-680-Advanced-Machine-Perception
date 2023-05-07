import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py
from collections import Counter
from matplotlib import patches
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer

from dataset import HW3DataModule
from model import SOLO

np.random.seed(17)

def main():
    trainer = Trainer(gpus=1, max_epochs=36, log_every_n_steps=100)
    datamodule = HW3DataModule()
    solo = SOLO().cuda()
    trainer.fit(solo, datamodule=datamodule)
    train_loss = np.array(solo.total_loss)
    cate_loss = np.array(solo.cate_loss)
    mask_loss = np.array(solo.mask_loss)
    valid_loss = np.array(solo.val_loss)
    np.save("total_loss.npy", train_loss)
    np.save("cate_loss.npy", cate_loss)
    np.save("mask_loss.npy", mask_loss)
    np.save( "valid_loss.npy", valid_loss)

if __name__ == "__main__":
    main()