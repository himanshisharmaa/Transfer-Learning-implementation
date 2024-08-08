from pyimg import config
from pyimg import create_dataset_loaders
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import shutil 
import torch
import time
import os


# define augmentation pipelines
trainTransform=transforms.Compose([
    transforms.RandomResizedCrop(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN,std=config.STD)
])

valTransform=transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN,std=config.STD)
])

# create data loaders

(trainDs,trainLoader)=create_dataset_loaders.get_dataloader(config.TRAIN,
                                                            transforms=trainTransform,
                                                            batchSize=config.FINE_TUNE_BATCH_SIZE,
                                                            shuffle=True)
(valDs,valLoader)=create_dataset_loaders.get_dataloader(
    config.VAL,
    transforms=valTransform,
    batchSize=config.FINE_TUNE_BATCH_SIZE,
    shuffle=False
)

model=resnet50(pretrained=True)
numFeatures=model.fc.in_features

# loop over the modules of the model and set the parameters of
# batch normalization modules as not trainable
for module,params in zip(model.modules(),model.parameters()):
    if isinstance(module,nn.BatchNorm2d):
        params.requires_grad=False

# define the network head and attach it to the model
headModel=nn.Sequential(
    nn.Linear(numFeatures,512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,len(trainDs.classes))
)

model.fc=headModel

#append a new classification top to our feature extractor and pop it
# on to the current device
model=model.to(config.DEVICE)

# initialize the loss function and optimizer(notice that we are only
# providing the parameters of the classification top to our optimizer)
lossFunc=nn.CrossEntropyLoss()
opt=torch.optim.Adam(model.parameters(),
                     lr=config.LR)

# calculate steps per epoch for training and validation set
trainSteps=len(trainDs)// config.FINE_TUNE_BATCH_SIZE
valSteps=len(valDs)//config.FINE_TUNE_BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [],
	"val_acc": []}

# loop over epochs
print("[INFO] training the network...")
starTime=time.time()
for e in tqdm(range(config.EPOCHS)):
    # set the model in training mode
    model.train()

    # total training and val loss
    totalTrainLoss=0
    totalValLoss=0

    #initialize the number of correct predictions
    trainCorrect=0
    valCorrect=0

    #loop over the training set
    for (i,(x,y)) in enumerate(trainLoader):
        # send the input to the device
        (x,y)=(x.to(config.DEVICE),y.to(config.DEVICE))

        # perform a forward pass and calculate the training loss
        pred=model(x)
        loss=lossFunc(pred,y)
        
        loss.backward()

        # check if we are updating the model parameters and if so
        #update them, and zero out the previously accumulated gradients
        if (i+2) %2==0:
            opt.step()
            opt.zero_grad()

        # add the loss to the total training loss so far and 
        # calculate the number of correct predictions
        totalTrainLoss+=loss
        trainCorrect+=(pred.argmax(1)==y).type(
            torch.float
        ).sum().item()
    
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # loop over validation set
        for (x,y) in valLoader:
            (x,y)=(x.to(config.DEVICE),
            y.to(config.DEVICE))

            pred=model(x)
            totalValLoss+=lossFunc(pred,y)

            # calculate the number of correct predictions
            valCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()

    # calculate the average training and validation
    avgTrainLoss= totalTrainLoss/trainSteps
    avgValLoss=totalValLoss/valSteps

    # calculate the training and validation accuracy
    trainCorrect=trainCorrect/len(trainDs)
    valCorrect=valCorrect/len(valDs)

    #update the training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H['train_acc'].append(trainCorrect)
    H['val_loss'].append(avgValLoss.cpu().detach().numpy())
    H['val_acc'].append(valCorrect)


    print(f"[INFO] {e+1}/{config.EPOCHS}")
    print(f"Training Loss: {avgTrainLoss:.4f}, Train Accuracy: {trainCorrect}")
    print(f"Val loss: {avgValLoss:.4f}, Val accuracy: {valCorrect}")


endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - starTime))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.FINETUNE_PLOT)

# serialize the model to disk
torch.save(model, config.FINETUNE_MODEL)