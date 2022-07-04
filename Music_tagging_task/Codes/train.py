"""
Description: Train file to train multi-label genre tagging
"""

import os
import torch
import argparse
import sys
import yaml
import model
import torch.nn as nn
from tqdm import tqdm
from torch import device, save
from data_loader import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
def get_all_dataloader():
    """
    dataloader to train nad validation dataset
    """
    train_loader = DataLoader(dataset=MyDataset(data_path, split='TRAIN'), 
                          batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(dataset=MyDataset(data_path, split='VALID'),
                          batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
   
    return train_loader, val_loader

def train_model(loader, optimizer, model):
    """
    input: train loader, optimizer, model
    return: loss
    """
    running_loss = 0
    model.train()
  
    for i, batch in enumerate(loader):
    
        optimizer.zero_grad()
        tag, spec,_ = batch
        out = model(spec.cuda())
        loss =  criterion(out, torch.tensor(tag).float().cuda())
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        running_loss += batch_loss
    

    return running_loss/len(loader)



def val_model(val_loader,model):
    """
    input: val loader, model
    return: loss
    """
    
    running_loss = 0
    model.eval()
    for i, batch in enumerate(val_loader):
        tag, spec, _ = batch
        out = model(spec.cuda())
        loss =  criterion(out, torch.tensor(tag).float().cuda())
        batch_loss = loss.item()
        running_loss += batch_loss
        

    return running_loss/len(val_loader)

if __name__ == '__main__':

    iter_name = sys.argv[1]
    writer = SummaryWriter(log_dir='./runs/'+iter_name)
    with open('config.yaml') as file: 
        config= yaml.safe_load(file)

    config = config["train"] 
   
    #### Parameters for training #####
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    learning_rate = config["learning_rate"]
    epochs = config["n_epochs"]
    
   

    ### Dataset Loader ##############
    data_path = config["data_path"]
    train_loader, val_loader = get_all_dataloader()
    checkpoints_path = config["checkpoints_path"]+iter_name + "/"
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)


    ##### Optimzers #################
    model = model.AudioModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    ########### Training ###############
    min_loss = 100
    for epoch in range(epochs):
        train_loss = train_model(train_loader, optimizer, model)
        val_loss_out =  val_model(val_loader,model)
        val_loss = val_loss_out
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Val Loss", val_loss, epoch)
        print("Epoch, Train, Val", epoch, train_loss, val_loss)
        if val_loss < min_loss:
            save(model.state_dict(), checkpoints_path + str(epoch) + ".pth")
            min_loss = val_loss
    