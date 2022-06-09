from train_triplet import BarlowModel
import pytorch_lightning as pl
import cv2
import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda")

model = nn.Sequential(
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
			nn.Linear(1024, 128)
		)

def cv2_img_to_tensor(cv2_img):
	"""
	takes in the opencv image, resizes it, normalizes it and returns the torch tensor
	:param cv2_img: opencv image
	:return: pytorch tensor, as a float, with channel [1, W, H, C]
	"""
	# resize the image
	img = cv2.resize(cv2_img, (128, 128), interpolation=cv2.INTER_LINEAR)
	# normalize
	img = img.astype(np.float32)/255.0
	# convert to torch tensor
	img = np.transpose(img, (2, 0, 1))
	# convert to float
	img = torch.from_numpy(img).unsqueeze(0).float()
	return img

# load the pytorch lightning model from checkpoint on CUDA
# model = BarlowModel(latent_size=64).load_from_checkpoint('models/07-06-2022-12-49-07_.ckpt', latent_size=128).to(device=device)

model.load_state_dict({k.replace('visual_encoder.',''):v for k,v in torch.load('models/07-06-2022-12-49-07_.ckpt')['state_dict'].items()})
model.to(device=device)

# read the image from disk [you might have other ways to get this image of a person - from detectron/yolo/etc]
img = cv2.imread('test/test.png', cv2.IMREAD_COLOR)

# convert the opencv image to tensor
tensor_img = cv2_img_to_tensor(img)

# run inference
descriptor = model(tensor_img.to(device=device))
# descriptor = model.visual_encoder(tensor_img.to(device=device))

print(descriptor.detach().cpu().numpy())