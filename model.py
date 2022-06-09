import torch.nn as nn

def get_model(latent_size=128):
	return nn.Sequential(
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
				nn.Linear(1024, latent_size)
			)