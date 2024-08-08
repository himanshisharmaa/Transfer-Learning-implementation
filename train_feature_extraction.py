from pyimg import config
from pyimg import create_dataset_loaders
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# define augmentation pipelines
trainTansform = transforms.Compose([
	transforms.RandomResizedCrop(config.IMAGE_SIZE),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
valTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])


# create data loaders
(trainDS, trainLoader) = create_dataset_loaders.get_dataloader(config.TRAIN,
	transforms=trainTansform,
	batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE)
(valDS, valLoader) = create_dataset_loaders.get_dataloader(config.VAL,
	transforms=valTransform,
	batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)


model=resnet50(pretrained=True)

# since we are using the ResNet50 model as a feature extractor we set
# its parameters to non-trainable (by default they are trainable)
for param in model.parameters():
    param.requires_grad=False

# append a new classification top to our feature extractor and pop it
# on to the current device
modelOutputFeats=model.fc.in_features
model.fc=nn.Linear(modelOutputFeats,len(trainDS.classes))
model=model.to(config.DEVICE)


# initialize loss function and optimizer (notice that we are only
# providing the parameters of the classification top to our optimizer)
lossFunc=nn.CrossEntropyLoss()
opt=torch.optim.Adam(model.fc.parameters(),lr=config.LR)

# calculate the steps per epoch for training and validation set
trainSteps=len(trainDS)// config.FEATURE_EXTRACTION_BATCH_SIZE
valSteps = len(valDS) // config.FEATURE_EXTRACTION_BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [],
	"val_acc": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.EPOCHS)):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFunc(pred, y)

		# calculate the gradients
		loss.backward()

		# check if we are updating the model parameters and if so
		# update them, and zero out the previously accumulated gradients
		if (i + 2) % 2 == 0:
			opt.step()
			opt.zero_grad()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()

	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFunc(pred, y)

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDS)
	valCorrect = valCorrect / len(valDS)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avgValLoss, valCorrect))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

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
plt.savefig(config.WARMUP_PLOT)

# serialize the model to disk
torch.save(model, config.WARMUP_MODEL)