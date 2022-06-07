import glob

import scipy.io
import numpy as np
import cv2
import h5py
import mat73
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import tensorflow as tf
import tensorboard as tb
import torchvision.transforms as transforms
# import resnet18
from torchvision.models import resnet18

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import List, Union, Tuple
from datetime import datetime
import random
from PIL import Image, ImageOps, ImageFilter

# # read .mat file
# mat = mat73.loadmat('RAID/RAiD_4Cams.mat')
#
# print(mat['dataset'].keys())
#
# images = mat['dataset']['images']
# masks = mat['dataset']['masks']
# personID = mat['dataset']['personID']
# cam = mat['dataset']['cam']
#
# print(cam[:100])
# print(personID[:100])
# print(images.shape)

# img = cv2.resize(images[:, :, :, 0], (128, 128), interpolation=cv2.INTER_CUBIC)
#
# cv2.imshow('image', img)
# cv2.imshow('mask', masks[:, :, 0])
# cv2.waitKey(0)

class GaussianBlur(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			sigma = random.random() * 1.9 + 0.1
			return img.filter(ImageFilter.GaussianBlur(sigma))
		else:
			return im

class Solarization(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if random.random() < self.p:
			return ImageOps.solarize(img)
		else:
			return img

class FacesDataset(Dataset):
	def __init__(self, data_path: str, train: bool = True):
		images_paths = glob.glob(data_path+'/*/*.jpg')
		img_path_dict = {}
		for img_path in images_paths:
			name = img_path.split('/')[-2]
			if name not in img_path_dict: img_path_dict[name] = []
			img_path_dict[name].append(img_path)

		# delete keys with less than 2 images
		for key in list(img_path_dict.keys()):
			if len(img_path_dict[key]) < 2:
				del img_path_dict[key]

		# rename keys to integers
		for i, key in enumerate(list(img_path_dict.keys())):
			img_path_dict[i] = img_path_dict.pop(key)

		print('Number of persons:', len(img_path_dict))

		# retain 80% of the data for training
		if train:
			person_ids = list(img_path_dict.keys())
			for i, key in enumerate(person_ids):
				if i > len(person_ids) * 0.7:
					del img_path_dict[key]

			print('Training set size:', len(img_path_dict))
			print('{} persons found with {} or more faces'.format(len(img_path_dict), 2))
		else:
			person_ids = list(img_path_dict.keys())
			for i, key in enumerate(person_ids):
				if i <= len(person_ids) * 0.7:
					del img_path_dict[key]

			print('Validation set size:', len(img_path_dict))
			print('{} persons found with {} or more faces'.format(len(img_path_dict), 5))

		self.img_path_dict = img_path_dict

		self.images, self.personID = [], []
		for key in list(img_path_dict.keys()):
			self.images.extend(img_path_dict[key])
			self.personID.extend([key] * len(img_path_dict[key]))

		# if train:
		# 	self.images = self.images[:int(len(self.images) * 0.8)]
		# 	self.personID = self.personID[:int(len(self.personID) * 0.8)]
		# else:
		# 	self.images = self.images[int(len(self.images) * 0.8):]
		# 	self.personID = self.personID[int(len(self.personID) * 0.8):]

		print('{} images found'.format(len(self.images)))


		# self.idx_consider = int(len(img_path_dict) * 0.75)
		# self.idx = np.arange(len(img_path_dict))
		#
		# # get the index of the training set
		# if train:
		# 	self.idx = self.idx[:int(len(img_path_dict) * 0.75)]
		# else:
		# 	self.idx = self.idx[int(len(img_path_dict) * 0.75):]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		anchor_person_id = self.personID[idx]
		anchor_img = cv2.imread(self.images[idx])
		anchor_img = cv2.resize(anchor_img, (128, 128), interpolation=cv2.INTER_CUBIC)
		anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
		anchor_img = np.transpose(anchor_img, (2, 0, 1))
		anchor_img = anchor_img.astype(np.float32) / 255.0

		# get the positive image
		positive_person_id = anchor_person_id
		positive_img_path = random.choice(self.img_path_dict[positive_person_id])
		positive_img = cv2.imread(positive_img_path)
		positive_img = cv2.resize(positive_img, (128, 128), interpolation=cv2.INTER_CUBIC)
		positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)
		positive_img = np.transpose(positive_img, (2, 0, 1))
		positive_img = positive_img.astype(np.float32) / 255.0

		# get the negative image
		negative_idxs = list(self.img_path_dict.keys())
		negative_idxs.remove(anchor_person_id)
		negative_img_list, negative_idxs_list = [], []
		for _ in range(10):
			negative_idx = random.choice(negative_idxs)
			negative_img_path = random.choice(self.img_path_dict[negative_idx])
			negative_img = cv2.imread(negative_img_path)
			negative_img = cv2.resize(negative_img, (128, 128), interpolation=cv2.INTER_CUBIC)
			negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)
			negative_img = np.transpose(negative_img, (2, 0, 1))
			negative_img = negative_img.astype(np.float32) / 255.0
			negative_img_list.append(negative_img)
			negative_idxs_list.append(negative_idx)


		# curr_person_image_paths = self.img_path_dict[self.idx[idx]]
		#
		# # sample two different images from the same person
		# img_path1, img_path2 = np.random.choice(curr_person_image_paths, 2, replace=False)
		# anchor_img = cv2.imread(img_path1)
		# positive_img = cv2.imread(img_path2)
		#
		# negative_idxs = list(self.idx)
		# negative_idxs.remove(self.idx[idx])
		# negative_img_list = []
		# for _ in range(10):
		# 	negative_id = np.random.choice(negative_idxs)
		# 	negative_img = np.random.choice(self.img_path_dict[negative_id])
		# 	negative_img = cv2.imread(negative_img)
		# 	negative_img_list.append(negative_img)
		#
		# randomneg = np.random.choice(np.arange(10))
		# cv2.imshow('anchor, positive, negative', np.hstack([anchor_img, positive_img, negative_img_list[randomneg]]))
		# cv2.waitKey(0)
		#
		# # resize the images
		# anchor_img = cv2.resize(anchor_img, (128, 128), interpolation=cv2.INTER_CUBIC)
		# positive_img = cv2.resize(positive_img, (128, 128), interpolation=cv2.INTER_CUBIC)
		# for i in range(len(negative_img_list)):
		# 	negative_img_list[i] = cv2.resize(negative_img_list[i], (128, 128), interpolation=cv2.INTER_CUBIC)
		#
		# # normalize the images
		# anchor_img = anchor_img.astype(np.float32) / 255.0
		# positive_img = positive_img.astype(np.float32) / 255.0
		# for i in range(len(negative_img_list)):
		# 	negative_img_list[i] = negative_img_list[i].astype(np.float32) / 255.0
		#
		# # change the channel order to (channel, height, width)
		# anchor_img = np.transpose(anchor_img, (2, 0, 1))
		# positive_img = np.transpose(positive_img, (2, 0, 1))
		# for i in range(len(negative_img_list)):
		# 	negative_img_list[i] = np.transpose(negative_img_list[i], (2, 0, 1))
		#
		# return anchor_img, positive_img, negative_img_list, self.idx[idx]
		return anchor_img, positive_img, negative_img_list, anchor_person_id, negative_idxs_list


class PersonDataset(Dataset):
	def __init__(self, data_path='/home/haresh/PycharmProjects/personReid/RAID/RAiD_4Cams.mat', train=True):
		mat = mat73.loadmat(data_path)

		self.idx_consider = int(len(mat['dataset']['personID']) * 0.75)

		self.idx = np.arange(len(mat['dataset']['personID']))

		# randomize the index
		np.random.shuffle(self.idx)

		# get the index of the training set
		self.idx_train = self.idx[:self.idx_consider]
		# get the index of the test set
		self.idx_test = self.idx[self.idx_consider:]

		# train set
		if train:
			self.images = mat['dataset']['images'][:, :, :, self.idx_train]
			self.personID = mat['dataset']['personID'][self.idx_train]
		else:
			self.images = mat['dataset']['images'][:, :, :, self.idx_test]
			self.personID = mat['dataset']['personID'][self.idx_test]

		self.personiddict = {}
		for i, personID in enumerate(self.personID):
			if personID not in self.personiddict.keys(): self.personiddict[personID] = []
			self.personiddict[personID].append(self.images[:, :, :, i])
		print('Number of datapoints : ', len(self.personID))

	def __len__(self):
		return len(self.personID)

	def __getitem__(self, idx):
		personID = self.personID[idx]

		# get the images of the same person
		positive_images = self.personiddict[personID]
		# randomly select one image
		positive_image = positive_images[np.random.randint(len(positive_images))]

		# get the images of other people
		negative_idxs = list(self.personiddict.keys())
		negative_idxs.remove(personID)

		negative_img_list = []
		for i in range(20):
			# randomly select a negative person from the list
			negative_images = self.personiddict[np.random.choice(negative_idxs)]
			# randomly select one image of that negative person
			negative_image = negative_images[np.random.randint(len(negative_images))]
			negative_img_list.append(negative_image)

		# current image
		anchor_image = self.images[:, :, :, idx]

		# resize the images
		anchor_image = cv2.resize(anchor_image, (128, 128), interpolation=cv2.INTER_CUBIC)
		positive_image = cv2.resize(positive_image, (128, 128), interpolation=cv2.INTER_CUBIC)
		for i in range(len(negative_img_list)):
			negative_img_list[i] = cv2.resize(negative_img_list[i], (128, 128), interpolation=cv2.INTER_CUBIC)
			# negative_image = cv2.resize(negative_image, (128, 128), interpolation=cv2.INTER_CUBIC)

		# normalize the images
		anchor_image = anchor_image.astype(np.float32) / 255.0
		positive_image = positive_image.astype(np.float32) / 255.0
		for i in range(len(negative_img_list)):
			negative_img_list[i] = negative_img_list[i].astype(np.float32) / 255.0
			# negative_image = negative_image.astype(np.float32) / 255.0

		# change the channel order
		anchor_image = np.transpose(anchor_image, (2, 0, 1))
		positive_image = np.transpose(positive_image, (2, 0, 1))
		for i in range(len(negative_img_list)):
			negative_img_list[i] = np.transpose(negative_img_list[i], (2, 0, 1))
			# negative_image = np.transpose(negative_image, (2, 0, 1))

		return anchor_image, positive_image, negative_img_list, personID

class FacesDataLoader(pl.LightningDataModule):
	def __init__(self, data_path='/home/haresh/PycharmProjects/personReid/lfw', batch_size=32):
		super(FacesDataLoader, self).__init__()
		self.data_path = data_path
		self.batch_size = batch_size

		self.train_dataset = FacesDataset(data_path=self.data_path, train=True)
		self.val_dataset = FacesDataset(data_path=self.data_path, train=False)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,
						  drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1,
						  drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)



class MyDataLoader(pl.LightningDataModule):
	def __init__(self, data_path='/home/haresh/PycharmProjects/personReid/RAID/RAiD_4Cams.mat', batch_size=32):
		super(MyDataLoader, self).__init__()
		self.data_path = data_path
		self.batch_size = batch_size

		self.train_dataset = PersonDataset(data_path=self.data_path, train=True)
		self.val_dataset = PersonDataset(data_path=self.data_path, train=False)


	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
						  drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
						  drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)


class BarlowModel(pl.LightningModule):
	def __init__(self, lr=3e-4, latent_size=64, scale_loss: float = 1.0 / 32, lamda: float = 3.9e-6, weight_decay=1e-6,
				 per_device_batch_size=32):
		super(BarlowModel, self).__init__()
		self.lr = lr
		self.latent_size = latent_size
		self.scale_loss = scale_loss
		self.lamda = lamda
		self.weight_decay = weight_decay
		self.per_device_batch_size = per_device_batch_size

		self.visual_encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False),  # 63 x 63
			nn.BatchNorm2d(16), nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),  # 31 x 31
			nn.BatchNorm2d(32), nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False),  # 14 x 14
			nn.BatchNorm2d(64), nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False),  # 5 x 5
			nn.BatchNorm2d(128), nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2 x 2
			nn.ReLU(),
			nn.Flatten(),  # 1024 output
			nn.Linear(1024, self.latent_size)
		)

		# use resnet18 encoder
		# self.visual_encoder = resnet18(pretrained=True)
		# self.visual_encoder.fc = nn.Linear(512, self.latent_size)


		self.triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2, swap=True)

		self.transform = transforms.Compose([
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(
				[transforms.ColorJitter(brightness=0.4, contrast=0.4,
										saturation=0.2, hue=0.1)],
				p=0.8
			),
			transforms.RandomGrayscale(p=0.2),
		])

	def forward(self, a, p, n):
		a_encoded = self.visual_encoder(a)
		p_encoded = self.visual_encoder(p)
		n_encoded = self.visual_encoder(n)

		# normalize the encoded vectors
		a_encoded = F.normalize(a_encoded, p=2, dim=1)
		p_encoded = F.normalize(p_encoded, p=2, dim=1)
		n_encoded = F.normalize(n_encoded, p=2, dim=1)

		loss = self.triplet_loss(a_encoded, p_encoded, n_encoded)
		return loss

	def training_step(self, batch, batch_idx):
		a, p, n, _ = batch
		# a = self.transform(a)
		# p = self.transform(p)
		# n = [self.transform(x) for x in n]

		# a_img = a[0].detach().cpu().numpy()
		# p_img = p[0].detach().cpu().numpy()
		# n_img = n[0][0].detach().cpu().numpy()
		# a_img = a_img.transpose(1, 2, 0)
		# p_img = p_img.transpose(1, 2, 0)
		# n_img = n_img.transpose(1, 2, 0)
		# print(a_img.shape)
		# cv2.imshow('a', np.hstack((a_img, p_img, n_img)))
		# cv2.waitKey(0)

		loss = 0
		for i in range(len(n)):
			loss += self.forward(a, p, n[i])
		self.log('train_loss', loss, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		a, p, n, _ = batch
		loss = 0
		for i in range(len(n)):
			loss += self.forward(a, p, n[i])
		self.log('val_loss', loss, prog_bar=True, logger=True)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):

		if self.current_epoch % 10 == 0:
			a, p, n, personID = batch
			with torch.no_grad():
				z1 = self.visual_encoder(a.cuda())
				z2 = self.visual_encoder(p.cuda())

				# normalize
				z1 = F.normalize(z1, p=2, dim=1)
				z2 = F.normalize(z2, p=2, dim=1)

			if batch_idx == 0:
				self.encoding = torch.cat((z1[:, :], z2[:, :]), dim=0)
				self.image = torch.cat((a[:, :, :, :], p[:, :, :, :]), dim=0)
				self.label = torch.cat((personID.cpu(), personID.cpu()), dim=0)
			else:
				self.encoding = torch.cat((self.encoding, torch.cat((z1[:, :], z2[:, :]), dim=0)), dim=0)
				self.image = torch.cat((self.image, torch.cat((a[:, :, :, :], p[:, :, :, :]), dim=0)), dim=0)
				self.label = np.concatenate((self.label, torch.cat((personID.cpu(), personID.cpu()), dim=0)), axis=0)

	def on_validation_end(self) -> None:
		if self.current_epoch % 10 == 0:
			self.logger.experiment.add_embedding(mat=self.encoding, label_img=self.image,
												 global_step=self.current_epoch, metadata=self.label)

	@property
	def total_training_steps(self) -> int:
		dataset_size = len(self.trainer.datamodule.train_dataloader())
		num_devices = self.trainer.tpu_cores if self.trainer.tpu_cores else self.trainer.num_processes
		effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
		max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

		if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
			return self.trainer.max_steps
		return max_estimated_steps

	def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
		return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_training_steps


if __name__ == '__main__':
	# parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--weight_decay', type=float, default=3e-6)
	parser.add_argument('--lamda', type=float, default=0.0051)
	parser.add_argument('--max_epochs', type=int, default=2000)
	parser.add_argument('--data_dir', type=str, default='/home/haresh/PycharmProjects/personReid/RAID/RAiD_4Cams.mat',
						metavar='N', help='data directory (default: data)')
	# parser.add_argument('--data_dir', type=str, default='/home/haresh/PycharmProjects/personReid/lfw',
	# 					metavar='N', help='data directory (default: data)')
	parser.add_argument('--log_dir', type=str, default='logs/', metavar='N',
						help='log directory (default: logs)')
	parser.add_argument('--model_dir', type=str, default='models/', metavar='N',
						help='model directory (default: models)')
	parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
						help='number of GPUs to use (default: 1)')
	parser.add_argument('--latent_size', type=int, default=128, metavar='N',
						help='Size of the common latent space (default: 128)')
	parser.add_argument('--use_faces', action='store_true', default=False)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# set the dataloader
	if args.use_faces:
		print('Using faces dataset')
		dm = FacesDataLoader(data_path=args.data_dir, batch_size=args.batch_size)
	else:
		print('Using full body dataset')
		dm = MyDataLoader(data_path=args.data_dir, batch_size=args.batch_size)
	model = BarlowModel(lr=args.lr, latent_size=args.latent_size, scale_loss=1.0, lamda=args.lamda,
						weight_decay=args.weight_decay, per_device_batch_size=args.batch_size).to(device)

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)
	model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
										  filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_',
										  monitor='val_loss', verbose=True)

	print("Training model...")
	trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
						 max_epochs=args.max_epochs,
						 callbacks=[model_checkpoint_cb],
						 log_every_n_steps=10,
						 distributed_backend='ddp',
						 num_sanity_val_steps=-1,
						 logger=True,
						 sync_batchnorm=True,
						 gradient_clip_val=1.0,
						 gradient_clip_algorithm='norm'
						 )

	trainer.fit(model, dm)