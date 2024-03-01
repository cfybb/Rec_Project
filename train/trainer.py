"""
An example of trainer: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""
import sys
from tqdm import tqdm

sys.path.append('C:/prdue/job_preperation_general/support_company/project/Rec_Project')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import unet_model
from data import pretrain
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split
# initilization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
batch_size = 24
learning_rate = 0.001
writer = SummaryWriter()

full_dataset = pretrain.CustomDataset()

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrain.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pretrain.collate_fn)

# initialization
model = unet_model.UNet(n_channels=3, n_classes=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss(reduction="sum")

# train and validate
num_step = 0
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    # train
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        images, labels, masks = batch['images'].to(device), batch['heatmaps'].to(device), batch["masks"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs * masks, labels * masks)
        print(f"loss: {loss}")
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_step += 1
        writer.add_scalar("Loss/train", loss.item(), num_step)

    train_loss /= len(train_dataloader)
    # validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            images, labels, masks = batch['images'].to(device), batch['heatmaps'].to(device), batch["masks"].to(device)
            outputs = model(images)
            loss = criterion(outputs * masks, labels * masks)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    writer.add_scalar("Loss/val", val_loss, num_step)

    # save model
    model_name = f"model_epoch_{epoch}.th"
    torch.save(model, model_name)
    print(f"Model saved to {model_name}...")

    # print
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
writer.flush()

print('Training Finished.')