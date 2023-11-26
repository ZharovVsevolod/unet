# import the necessary packages
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import cv2
import kaggle
from . import config
import os
import shutil
import lightning as L
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from imutils import paths

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms

		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]

		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)

		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)

		# return a tuple of the image and its mask
		return (image, mask)

def load_data():
	# os.environ['KAGGLE_USERNAME'] = "alagril"
	# os.environ['KAGGLE_KEY'] = "19e923d361a864eb90e12d51d518b7f1"
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files(dataset="mnpinto/pneumotorax128", path=config.DATASET_PATH, unzip=True)

	# Поскольку он почему-то дублирует файлы (и вообще очень странно их скачивает), мы их удаляем
	shutil.move(os.path.join(config.DATASET_PATH, "data128", "masks"), config.DATASET_PATH)
	shutil.move(os.path.join(config.DATASET_PATH, "data128", "test"), config.DATASET_PATH)
	shutil.move(os.path.join(config.DATASET_PATH, "data128", "train"), config.DATASET_PATH)
	shutil.rmtree(os.path.join(config.DATASET_PATH, "data128"))

class Unet_DataModule(L.LightningDataModule):
	def __init__(self) -> None:
		super().__init__()
	
	def load_data(self):
		# os.environ['KAGGLE_USERNAME'] = "alagril"
		# os.environ['KAGGLE_KEY'] = "19e923d361a864eb90e12d51d518b7f1"
		kaggle.api.authenticate()
		kaggle.api.dataset_download_files(dataset="mnpinto/pneumotorax128", path=config.DATASET_PATH, unzip=True)

		# Поскольку он почему-то дублирует файлы (и вообще очень странно их скачивает), мы их удаляем
		shutil.move(os.path.join(config.DATASET_PATH, "data128", "masks"), config.DATASET_PATH)
		shutil.move(os.path.join(config.DATASET_PATH, "data128", "test"), config.DATASET_PATH)
		shutil.move(os.path.join(config.DATASET_PATH, "data128", "train"), config.DATASET_PATH)
		shutil.rmtree(os.path.join(config.DATASET_PATH, "data128"))
	
	def prepare_data(self) -> None:
		if not os.path.isdir(config.IMAGE_DATASET_PATH):
			self.load_data()
		else:
			print("Данные уже загружены")

	def setup(self, stage: str) -> None:
		# define transformations
		transforming = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
			transforms.ToTensor()
		])

		if stage == "fit" or stage is None:
			# load the image and mask filepaths in a sorted manner
			imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
			maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

			# partition the data into training and testing splits using 85% of
			# the data for training and the remaining 15% for testing
			trainImages, testImages, trainMasks, testMasks = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)
			
			# create the train and test datasets
			self.trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforming)
			self.testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforming)

		if stage == "test" or stage is None:
			testPaths = sorted(list(paths.list_images(config.TEST_DATASET_PATH)))
			self.testDataset = SegmentationDataset_for_Test(imagePaths=testPaths, transforms=transforming)

	def train_dataloader(self) -> TRAIN_DATALOADERS:
		return DataLoader(
			self.trainDS, 
			shuffle=True,
			batch_size=config.BATCH_SIZE
		)
	
	def val_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(
			self.testDS, 
			shuffle=False,
			batch_size=config.BATCH_SIZE
		)
	
	def test_dataloader(self) -> EVAL_DATALOADERS:
		return DataLoader(
			self.testDataset, 
			shuffle=False,
			batch_size=config.BATCH_SIZE
		)

class SegmentationDataset_for_Test(Dataset):
	def __init__(self, imagePaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms

		self.imagePaths = imagePaths
		self.transforms = transforms

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]

		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
		# return a tuple of the image and its mask
		return image