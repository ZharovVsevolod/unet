# import the necessary packages
import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT, LRSchedulerPLType, OptimizerLRScheduler
from . import config
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torch import nn
from typing import Any, List, Literal, Sequence, Union
import einops
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class Block(nn.Module):
	def __init__(self, in_channels, out_channels=None, mode:Literal["down", "up"]="down"):
		super().__init__()
		# store the convolution and RELU layers
		if not (mode in ["down", "up"]):
			raise Exception("Не тот тип mode", mode)

		if out_channels is None:
			match mode:
				case "down":
					out_channels = in_channels // 2
				case "up":
					out_channels = in_channels * 2
				case _:
					out_channels = in_channels
		
		self.convblock = nn.Sequential(
			nn.Conv2d(
				in_channels = in_channels,
				out_channels = out_channels,
				kernel_size = 3
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels = out_channels,
				out_channels = out_channels,
				kernel_size = 3
			),
			nn.ReLU()
		)

	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		x = self.convblock(x)
		return x

class Encoder(nn.Module):
	def __init__(self, channels=[3, 16, 32, 64]):
		super().__init__()

		# store the encoder blocks and maxpooling layer
		self.enc_blocks = nn.ModuleList([
			Block(
				in_channels=channels[i],
				out_channels=channels[i+1]
			)
			for i in range(len(channels) - 1)
		])

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		block_outputs = []

		# loop through the encoder blocks
		for block in self.enc_blocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			block_outputs.append(x)
			x = self.pool(x)

		# return the list containing the intermediate outputs
		return block_outputs

class Decoder(nn.Module):
	def __init__(self, channels=[64, 32, 16]):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels

		self.upconvs = nn.ModuleList([
			nn.ConvTranspose2d(
				in_channels=channels[i],
				out_channels=channels[i+1],
				kernel_size=2,
				stride=2
			)
			for i in range(len(channels) - 1)
		])

		self.dec_blocks = nn.ModuleList([
			Block(
				in_channels=channels[i],
				out_channels=channels[i+1]
			)
			for i in range(len(channels) - 1)
		])

	def forward(self, x, enc_features):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			croped_enc_features = self.crop(enc_features[len(self.channels) - i - 2], x)
			croped_enc_features = einops.rearrange(croped_enc_features, "b c h w -> b c h w", c=x.shape[1]) # Проверка размерности
			x = torch.cat((x, croped_enc_features), dim=1)
			x = einops.rearrange(x, "b c h w -> b c h w", c=self.channels[i]) # Проверка размерности
			x = self.dec_blocks[i](x)

		# return the final decoder output
		return x

	def crop(self, enc_features, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		enc_features = CenterCrop([H, W])(enc_features)

		# return the cropped features
		return enc_features

class UNet(nn.Module):
	def __init__(
			self, 
			enc_channels=[config.NUM_CHANNELS, 16, 32, 64], 
			dec_channels=[64, 32, 16],
			n_classes=config.NUM_CLASSES, 
			retain_dim=True,
			out_size=[config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH]
		):
		super().__init__()

		# initialize the encoder and decoder
		self.encoder = Encoder(enc_channels)
		self.decoder = Decoder(dec_channels)

		# initialize the regression head and store the class variables
		self.head = nn.Conv2d(
			in_channels=dec_channels[-1],
			out_channels=n_classes,
			kernel_size=1
		)
		self.retain_dim = retain_dim
		self.out_size = out_size

	def forward(self, x):
		# grab the features from the encoder
		enc_features = self.encoder(x)

		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		dec_features = self.decoder(enc_features[-1], enc_features[:-1])

		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(dec_features)

		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retain_dim:
			map = F.interpolate(map, self.out_size)

		# return the segmentation map
		return map

class Unet_lightning(L.LightningModule):
	def __init__(
			self,
			previous_model = None,
			enc_channels=[config.NUM_CHANNELS, 16, 32, 64],
			dec_channels=[64, 32, 16],
			n_classes=config.NUM_CLASSES,
			retain_dim=True,
			out_size=[config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH]
		) -> None:
		super().__init__()
		if previous_model is None:
			self.unet_model = UNet(
				enc_channels=enc_channels,
				dec_channels=dec_channels,
				n_classes=n_classes,
				retain_dim=retain_dim,
				out_size=out_size
			)
		else:
			self.add_module = previous_model
		self.save_hyperparameters()
	
	def forward(self, x) -> Any:
		return self.unet_model(x)
	
	def loss(self, x, y):
		return F.binary_cross_entropy_with_logits(x, y)

	def training_step(self, batch) -> STEP_OUTPUT:
		x, y = batch
		out = self(x)
		pred_loss = self.loss(out, y)
		self.log("train_loss", pred_loss)
		return pred_loss
	
	def validation_step(self, batch) -> STEP_OUTPUT:
		x, y = batch
		out = self(x)
		pred_loss = self.loss(out, y)
		self.log("val_loss", pred_loss)
	
	def test_step(self, batch) -> STEP_OUTPUT:
		x = batch
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = self(x).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
		ax[0].imshow(predMask)
		return figure
	
	def configure_optimizers(self) -> OptimizerLRScheduler:
		opt = torch.optim.Adam(self.unet_model.parameters(), lr=config.INIT_LR)
		return opt
	
class LightCheckPoint(L.Callback):
	def __init__(self, logger=None) -> None:
		super().__init__()

		self.logger = logger

		self.test_imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
	
	def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		torch.save(pl_module.unet_model, config.MODEL_PATH)
	
	def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
		path_test_image = np.random.choice(self.test_imagePaths, size=config.TEST_IMAGES_SIZE)
		some_images = []
		for path_img in path_test_image:
			fig = self.make_pred(path_img, pl_module)
			some_images.append(fig)
		if self.logger is not None:
			self.logger.log_image(key="Test mask prediction", images=some_images)
	
	def make_pred(self, imagePath, model: L.LightningModule):
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()

		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)

		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))

  		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		# prepare a plot for visualization
		fig = self.prepare_plot(orig, gtMask, predMask)
		return fig
	
	def prepare_plot(self, origImage, origMask, predMask):
		# initialize our figure
		figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

		# plot the original image, its mask, and the predicted mask
		ax[0].imshow(origImage)
		ax[1].imshow(origMask)
		ax[2].imshow(predMask)

		# set the titles of the subplots
		ax[0].set_title("Image")
		ax[1].set_title("Original Mask")
		ax[2].set_title("Predicted Mask")

		# set the layout of the figure and display it
		figure.tight_layout()
		return figure