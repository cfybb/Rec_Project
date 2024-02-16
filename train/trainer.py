"""
An example of trainer: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""
import sys

sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import unet_model
from data import pretrain


# initilization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
batch_size = 4
learning_rate = 0.001


full_dataset = pretrain.CustomDataset()

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrain.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# initialization
model = unet_model.UNet(n_channels=3, n_classes=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# train and validate
for epoch in range(num_epochs):
    # train
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch in train_dataloader:
        print(type(batch))
        images, labels = batch["images"].to(device), batch["heatmaps"].to(device)  # 0 for iamge, 1 for heatmap
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # train accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataloader)
    train_accuracy = 100 * correct / total

    # validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images, labels = batch['images'].to(device), batch['heatmaps'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # validate accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = 100 * val_correct / val_total

    # print
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

print('Training Finished.')