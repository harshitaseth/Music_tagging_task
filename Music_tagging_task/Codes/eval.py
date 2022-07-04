"""File use to evaluate the model on validation dataset.
"""
import os
import torch
import sys
import model
import yaml
import pickle
import argparse
import numpy as np
import torch.nn as nn
from sklearn import metrics
from torch import device, save
from data_loader import MyDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
def get_all_dataloader():
    
    val_loader = DataLoader(dataset=MyDataset(data_path, split='VALID'),
                          batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
   
    return val_loader




def get_evaluation(val_loader,model):
    """
    input: dataset loader, model
    apply sigmoid on model output
    compute precision using top 10 predicted tags

    """
    p_ks = []
    output_dict = {}
    sigmoid = nn.Sigmoid()
    
    for i, batch in enumerate(val_loader):
        tag, spec, name = batch
        output_dict[name] = {}
        out = model(spec.cuda())
        # putting sigmoid over last layer of model. During training we have used BCEWithLogitsLoss
        out = sigmoid(out)
        # sorting value and take top 10
        output = np.argsort(out.cpu().detach().numpy())[0][::-1][:10]
        labels_out= []
        for val in output:
            labels_out.append(reverse_labels[val])
        pred = np.zeros(len(tag[0]))
        pred[output] = 1
        # calculation precison on single file
        p_k = metrics.precision_score(tag[0], pred)
        p_ks.append(p_k)
        output_dict[name]["predicted"] = labels_out
        output_dict[name]["metric: precision@10"] = p_k
    
    print("Precision", sum(p_ks)/len(p_ks))
        

if __name__ == '__main__':


    
    with open('config.yaml') as file: 
        config= yaml.safe_load(file)

    config = config["val"] 
    data_path = config["data_path"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"] 
    labels = pickle.load(open(os.path.join(data_path,"label_dict.pkl"), 'rb'))
    # converting labels back to tag to save in output file
    reverse_labels = {}
    for key  in labels.keys():
        reverse_labels[labels[key]] = key
    is_balanced = False
   

    val_loader = get_all_dataloader()

    model = model.AudioModel().cuda()
    model_path = config["checkpoints_path"]
    model.load_state_dict(torch.load(model_path))
    model.eval()
    get_evaluation(val_loader,model)
    

