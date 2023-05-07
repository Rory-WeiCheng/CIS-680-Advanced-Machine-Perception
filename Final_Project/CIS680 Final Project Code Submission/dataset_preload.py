from torch.utils import data
from PIL import Image
import numpy as np
import torch
import glob
from tqdm import tqdm

class Edge2Shoe(data.Dataset):
	""" Dataloader for Edge2Shoe datasets 
		Note: we resize images (original 256x256) to 128x128 for faster training purpose 
		
		Args: 
			img_dir: path to the dataset

	"""
	def __init__(self, img_dir, img_dim=128):
		# if img_dir == 'train':
		# 	self.img = np.load('data/train_img_30k.npz',allow_pickle=True)['arr_0']
		# else:
		# 	self.img = np.load('data/val_img.npz',allow_pickle=True)['arr_0']
		self.img_dim = img_dim
		self.img = np.load(img_dir, allow_pickle=True)['arr_0']


	def __getitem__(self, index):
		image_tensor = self.img[index]
		edge_tensor = image_tensor[:,:,:self.img_dim]; rgb_tensor = image_tensor[:,:,self.img_dim:]
		return edge_tensor, rgb_tensor


	def __len__(self):
		return len(self.img)


if __name__ == '__main__':
	img_dir_train = 'data/edges2shoes/train/' 
	train_dataset = Edge2Shoe(img_dir_train)
	loader = data.DataLoader(train_dataset, batch_size=32)
	for idx, batch in enumerate(loader):
		edge_tensor, rgb_tensor = batch
		print(idx, edge_tensor.shape, rgb_tensor.shape) # (32,3,128,128)
		break
