_exp_name = "resnet"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
from torchvision.models import resnet50
from torchvision.models import vgg16  # Import VGG16 model

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 256)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally.
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),  # Randomly rotate the image by 15 degrees.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation by 10%.
    transforms.RandomCrop(128, padding=4),  # Crop randomly with padding.
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur with random strength.

    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])




class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im,label
    

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(VGGClassifier, self).__init__()
        self.vgg = vgg16(pretrained=False)
        # Replace the classifier to adapt to the number of classes in your dataset
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)
    
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet50(pretrained=False)
        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dim [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            nn.Dropout(0.3), 

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.3), 

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.4), 

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.4), 

            nn.Conv2d(512, 1024, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.5), 

        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = ResNetClassifier().to(device)
model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
# model = VGGClassifier(num_classes=11).to(device)

# model = ResNetClassifier().to(device)
# model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))

# The number of batch size.
batch_size = 64

# The number of training epochs.
n_epochs = 8

# If no improvement in 'patience' epochs, early stop.
patience = 5

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset("./train", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset("./valid", tfm=train_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# Merge train and valid datasets.
merged_dataset = ConcatDataset([train_set, valid_set])

# Create a DataLoader for the merged dataset.
merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

'''
# Generate shuffled indices
shuffled_indices = torch.randperm(len(merged_dataset))

# Create a new dataset with shuffled indices
shuffled_dataset = Subset(merged_dataset, shuffled_indices)

# Create a DataLoader for the shuffled dataset
shuffled_loader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
'''


n_folds = 8
all_indices = np.arange(len(merged_dataset))
folds = np.array_split(all_indices, n_folds)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    fold_losses = []
    fold_accuracies = []

    for fold in range(n_folds):

        train_idx = [idx for i, indices in enumerate(folds) if i != fold for idx in indices]
        valid_idx = folds[fold]

        train_dataset = Subset(merged_dataset, train_idx)
        valid_dataset = Subset(merged_dataset, valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        train_losses = []
        train_accs = []

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_losses.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Cross Train | {epoch + 1:03d}/{n_epochs:03d}, {fold + 1:03d}/{n_folds:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_losses = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_losses.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_losses) / len(valid_losses)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Cross Valid | {epoch + 1:03d}/{n_epochs:03d}, {fold + 1:03d}/{n_folds:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        valid_acc_tta = valid_acc
        valid_loss_tta = valid_loss

        fold_losses.append(valid_loss_tta)
        fold_accuracies.append(valid_acc_tta)

    avg_valid_loss = sum(fold_losses) / len(fold_losses)
    avg_valid_acc = sum(fold_accuracies) / len(fold_accuracies)

    # update logs
    if avg_valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {avg_valid_loss:.5f}, acc = {avg_valid_acc:.5f}")


    # save models
    if avg_valid_acc > best_acc:
        print(f"Best model found at epoch {epoch+1}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = avg_valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break


# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("./test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


model_best = ResNetClassifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission_1.csv",index = False)