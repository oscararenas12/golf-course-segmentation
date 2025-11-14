# %%

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
jacotaco_danish_golf_courses_orthophotos_path = kagglehub.dataset_download('jacotaco/danish-golf-courses-orthophotos')

print('Data source import complete.')
print(f"Dataset path: {jacotaco_danish_golf_courses_orthophotos_path}")


#%%
import os
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
import torchvision.transforms.functional as TF

#Plotting images
from PIL import Image
import matplotlib.pyplot as plt

# %% 
#Hyperparameters
BATCH_SIZE = 16 #Number of batches when training
IMAGE_SIZE = (256, 256)#(320, 192) #Images get resized to a smaller resolution
IN_CHANNELS = 3 #There are 3 channels for RGB
LEARNING_RATE = 1e-4

base_path = jacotaco_danish_golf_courses_orthophotos_path
IMAGES_DIR = os.path.join(base_path, '1. orthophotos')
SEGMASKS_DIR = os.path.join(base_path, '2. segmentation masks')
LABELMASKS_DIR = os.path.join(base_path, '3. class masks')


# %%
#Loading the data
orthophoto_list = os.listdir(IMAGES_DIR)
print("There are ", len(orthophoto_list), " orthophotos in this dataset!")

#Load image with index of 5 (I prefer this image as it shows all the classes)
idx = 5 #The index can be changed to view other orthophotos.
golf_image = Image.open(os.path.join(IMAGES_DIR, orthophoto_list[idx]))
golf_segmask = Image.open(os.path.join(SEGMASKS_DIR, orthophoto_list[idx].replace(".jpg", ".png"))) #The class masks are png instead of jpg

#Plot using matplotlib
fig, axes = plt.subplots(1, 2)

axes[0].set_title('Orthophoto')
axes[1].set_title('Segmentation Mask')

axes[0].imshow(golf_image)
axes[1].imshow(golf_segmask)

# %%
class GolfDataset(Dataset):
    def __init__(self, images_dir, labelmasks_dir):
        #The directories for each folder
        self.images_dir = images_dir
        self.labelmasks_dir = labelmasks_dir

        self.images_dir_list = os.listdir(images_dir) #We create a list of PATHs to every file in the orthophotos directory.

    def __len__(self):
        return len(self.images_dir_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images_dir_list[idx])
        image = read_image(image_path, ImageReadMode.RGB)

        label_mask_path = os.path.join(self.labelmasks_dir, self.images_dir_list[idx]).replace(".jpg", ".png") #The class masks are png instead of jpg
        label_mask = read_image(label_mask_path, ImageReadMode.GRAY)

        #Apply transformations to the images. This can be optimized using nn.Sequential or nn.Compose.
        image = TF.resize(image, IMAGE_SIZE) #Apply resize transform
        image = image.float()
        image = image / 255 #Normalize values from [0-255] to [0-1]

        label_mask = TF.resize(label_mask, IMAGE_SIZE) #Apply resize transform
        label_mask = TF.rgb_to_grayscale(label_mask) #Apply grayscaling to go from 3->1 channels.
        label_mask = label_mask.float()

        return image, label_mask
    
# %%
golf_ds = GolfDataset(IMAGES_DIR, LABELMASKS_DIR)
idx = 5
orthophoto = golf_ds.__getitem__(idx)[0]
label_mask = golf_ds.__getitem__(idx)[1]
print("Ortophoto: ", orthophoto.shape, orthophoto)
print("Label:", label_mask.shape, label_mask)

# %%
class GolfDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.all_images = []

    def prepare_data(self):
        #We don't use this function for loading the data as prepare_data is called from a single GPU.
        #It can also not be usedto assign state (self.x = y).
        pass

    def setup(self, stage=None):
        #Data is loaded from the image and mask directories
        self.all_images = GolfDataset(IMAGES_DIR, LABELMASKS_DIR)
        #The data is split into train, val and test with a 70/20/10 split
        self.train_data, self.val_data, self.test_data = random_split(self.all_images, [0.7,0.2,0.1])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
         return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
         return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2, pin_memory=True, persistent_workers=True)


# %%
class UNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #DoubleConvSame has padding=1 which keeps the input and ouput dimensions the same.
        class DoubleConvSame(nn.Module):
            def __init__(self, c_in, c_out):
                super(DoubleConvSame, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)

        self.conv1 = DoubleConvSame(c_in=3, c_out=64)
        self.conv2 = DoubleConvSame(c_in=64, c_out=128)
        self.conv3 = DoubleConvSame(c_in=128, c_out=256)
        self.conv4 = DoubleConvSame(c_in=256, c_out=512)
        self.conv5 = DoubleConvSame(c_in=512, c_out=1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.up_conv1 = DoubleConvSame(c_in=1024, c_out=512)
        self.up_conv2 = DoubleConvSame(c_in=512, c_out=256)
        self.up_conv3 = DoubleConvSame(c_in=256, c_out=128)
        self.up_conv4 = DoubleConvSame(c_in=128, c_out=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_loss = []
        self.val_loss = []

    def crop_tensor(self, up_tensor, target_tensor):
        _, _, H, W = up_tensor.shape

        x = T.CenterCrop(size=(H, W))(target_tensor)

        return x

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)


        c2 = self.conv2(p1)
        p2 = self.pool(c2)


        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        """BOTTLE-NECK"""

        c5 = self.conv5(p4)
        """DECODER"""

        u1 = self.up1(c5)
        crop1 = self.crop_tensor(u1, c4)
        cat1 = torch.cat([u1, crop1], dim=1)
        uc1 = self.up_conv1(cat1)

        u2 = self.up2(uc1)
        crop2 = self.crop_tensor(u2, c3)
        cat2 = torch.cat([u2, crop2], dim=1)
        uc2 = self.up_conv2(cat2)

        u3 = self.up3(uc2)
        crop3 = self.crop_tensor(u3, c2)
        cat3 = torch.cat([u3, crop3], dim=1)
        uc3 = self.up_conv3(cat3)

        u4 = self.up4(uc3)
        crop4 = self.crop_tensor(u4, c1)
        cat4 = torch.cat([u4, crop4], dim=1)
        uc4 = self.up_conv4(cat4)

        outputs = self.conv_1x1(uc4)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        _y = torch.squeeze(y).long() #Squeeze to go from (B, 1, H, W) to (B, H, W), and converted to dtype of long - Needed for cross entropy loss!

        loss = self.loss_fn(y_pred, _y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)
        _y = torch.squeeze(y).long() #Squeeze to go from (B, 1, H, W) to (B, H, W), and converted to dtype of long - Needed for cross entropy loss!

        loss = self.loss_fn(y_pred, _y)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        _y = torch.squeeze(y).long() #Squeeze to go from (B, 1, H, W) to (B, H, W), and converted to dtype of long - Needed for cross entropy loss!

        loss = self.loss_fn(y_pred, _y)

        save_predictions_as_imgs(x, y, y_pred, counter=batch_idx)

        return loss

    def test_epoch_end(self, outs):
        print("Testing ended!")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)


# %%
folder="/kaggle/working/"
def save_predictions_as_imgs(x, y, y_pred, counter=0):
    #Currently the groundtruth and prediction (y & y_pred) have the shape [B, C, H, W] = [B, 1, H, W].
    #If we save them as images it will be in grayscale as the number of channels is 1.
    #Therefor, we have to convert them to 3 channels (RGB), and each class gets their own color.
    y_in_rgb = torch.zeros(y.shape[0], 3, y.shape[2], y.shape[3]).to(y.device)

    y_pred_in_rgb = torch.zeros(y.shape[0], 3, y.shape[2], y.shape[3]).to(y.device)
    #Create a list of tensors containing the rgb colors for each class
    #The list is [Background, Fairway, Green, Tee, Bunker, Water]
    class_colors = [torch.tensor([0, 0, 0]).to(y.device), torch.tensor([0.0, 140.0/255, 0.0]).to(y.device), torch.tensor([0.0, 1.0, 0.0]).to(y.device), torch.tensor([1.0, 0.0, 0.0]).to(y.device), torch.tensor([217.0/255, 230.0/255, 122.0/255]).to(y.device), torch.tensor([7.0/255, 15.0/255, 247.0/255]).to(y.device)]

    y_pred = calculate_labels_from_pred(y_pred) #Converted the prediction (stored as probalities) to labels!
    for c in range(1, 6): #loop through the classes 1-5
        y_mask = torch.where(y == c, 1, 0).to(y.device)
        y_pred_mask = torch.where(y_pred == c, 1, 0).to(y_pred.device)
        current_class_color = class_colors[c].reshape(1, 3, 1, 1)
        y_segment = y_mask*current_class_color
        y_pred_segment = y_pred_mask*current_class_color
        y_in_rgb += y_segment
        y_pred_in_rgb += y_pred_segment

    #Save images to /kaggle/working/
    #The images can be downloaded, or visualized later with matplotlib!
    torchvision.utils.save_image(x, f"{folder}/{counter+1}_figure.jpg")
    torchvision.utils.save_image(y_in_rgb, f"{folder}/{counter+1}_groundtruth.jpg")
    torchvision.utils.save_image(y_pred_in_rgb, f"{folder}/{counter+1}_prediction.jpg")

softmax = nn.Softmax2d()
def calculate_labels_from_pred(pred):
    pred = softmax(pred)
    pred = torch.argmax(pred, dim=1)
    pred = pred.float()
    pred = pred.unsqueeze(1)
    pred.requires_grad_()
    return pred


# %%
train_loader = GolfDataModule(BATCH_SIZE)
trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=2, log_every_n_steps=24, strategy="ddp_notebook_find_unused_parameters_false")
model = UNetModel()


trainer.fit(model, train_loader)


# %%
# automatically loads the best weights for you
trainer = pl.Trainer(devices=1, num_nodes=1, accelerator='gpu')
trainer.test(model, train_loader)


# %%
output_dir = '/kaggle/working/'

#Load the latest images from the validation!
for idx in range(1, 8): #Show some of the batches
    orthophoto = Image.open(output_dir + str(idx) + '_figure.jpg')
    groundtruth = Image.open(output_dir + str(idx) + '_groundtruth.jpg')
    prediction = Image.open(output_dir + str(idx) + '_prediction.jpg')

    #Plot using matplotlib
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(18.5, 15.5)

    axes[0].set_title('Orthophoto')
    axes[1].set_title('Groundtruth')
    axes[2].set_title('Prediction')

    axes[0].imshow(orthophoto)
    axes[1].imshow(groundtruth)
    axes[2].imshow(prediction)