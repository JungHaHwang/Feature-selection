import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader, random_split
import os

os.makedirs("./models", exist_ok=True)
os.makedirs("./backbone_outputs", exist_ok=True)

# dataset definition
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()
        
        # 2D Convolutional Layers: input channels=1 (grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=0)
        
        
        self.batch1 = nn.BatchNorm2d(8)
        self.batch2 = nn.BatchNorm2d(16)
        self.batch3 = nn.BatchNorm2d(16)
        self.batch4 = nn.BatchNorm2d(8)
        self.batch5 = nn.BatchNorm2d(1)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.fc1 = nn.Linear(1 * 5 * 5, 2)

    def forward(self, x):
        
        # Convolution + Activation + Pooling
        x = self.pool1(F.relu(self.batch1(self.conv1(x))))
        x = self.pool2(F.relu(self.batch2(self.conv2(x))))
        x = self.pool3(F.relu(self.batch3(self.conv3(x))))
        x = self.pool4(F.relu(self.batch4(self.conv4(x))))
        x = self.pool5(F.relu(self.batch5(self.conv5(x))))
        
        # Flatten the tensor
        x = x.view(-1, 1 * 5 * 5)
        
        # Fully connected layers
        x = self.fc1(x)
        
        return x

train_data_path = "./dataset_baseline/train_data.npy"
train_label_path = "./dataset_baseline/train_labels.npy"
val_data_path = "./dataset_baseline/test_data.npy"
val_label_path = "./dataset_baseline/test_labels.npy"

train_data = np.load(train_data_path)
train_data = torch.from_numpy(train_data).float().unsqueeze(1)
train_data = train_data / 255.0

train_label = np.load(train_label_path)
train_label = torch.from_numpy(train_label).long()
train_dataset = MyDataset(train_data, train_label)

val_data = np.load(val_data_path)
val_data = torch.from_numpy(val_data).float().unsqueeze(1)
val_data = val_data / 255.0

val_label = np.load(val_label_path)
val_label = torch.from_numpy(val_label).long()
val_dataset = MyDataset(val_data, val_label)


# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
feature_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy


criterion = nn.CrossEntropyLoss()
num_epochs = 200   # maximum epochs

count = 0
label_saved = 0
for repeat in range(1000):

    if count == 50:
        print("model training is terminated")
        break 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN2DModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if train_acc == 100.0 and val_acc == 100.0:
            count += 1
            model_path = f"./models/parking_{repeat}.pth"
            torch.save(model, model_path)
            
            flattened_outputs = []

            def hook_fn(module, input, output):
                flattened_outputs.append(output.detach().cpu().numpy())

            hook = model.pool5.register_forward_hook(hook_fn)

            inputs_list = []
            labels_list = []


            with torch.no_grad():
                for inputs, labels in feature_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    inputs_list.append(inputs.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())

                    model(inputs)

            # save backbone outputs
            output_file_path = f"./backbone_outputs/outputs_{repeat}.npy"
            np.save(output_file_path, flattened_outputs)  

            # save labels corresponding to backbone outputs
            if label_saved == 0:
                label_file_path = f"./backbone_outputs/labels.npy"
                np.save(label_file_path, labels_list)
                label_saved += 1


            hook.remove()

            del model
            torch.cuda.empty_cache()
            gc.collect()
            break
