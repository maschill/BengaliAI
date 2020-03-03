from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm

from .datagen import BengaliGraphemes
from .models import densenet
from .models import densenet121
from .models import densenet169


IMG_SIZE = 128
BATCH_SIZE = 512 // IMG_SIZE * 64
NUM_EPOCHS = 50

def train_model(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        img_size=IMG_SIZE,
        model=None,
        lr_min=None,
        lr_max=None,
        clr=True,
        weighted_classes=False
    ):

    if lr_min is None: 
        lr_min = 1.0e-3
    
    if lr_max is None: 
        lr_max = 5.0e-3

    cycle_half_len = int(.4*num_epochs)
    cycle_len =  cycle_half_len*2
    lr_delta_adam = (lr_min-(lr_min*.99 / (num_epochs-cycle_len)))/lr_min

    dl = BengaliGraphemes(batch_size)
    num_train_batches = dl.sizes["train"]//batch_size+1
    device = torch.device("cuda")

    if model is None: 
        model = densenet(dl.num_classes, in_channels=1, net=121)

    model.to(device)

    # setup for model saving
    starttime = f"start-{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.mkdir(f"models/{starttime}")
    weighted_classes_name = "notweighted"
    optimizer_type = "Linear"

    # # setup weighted loss and optimizer
    if weighted_classes:
        weighted_classes_name = "weighted-loss"
        grapheme_weights = torch.FloatTensor(dl.weights["grapheme"]).cuda()
        vowel_weights = torch.FloatTensor(dl.weights["vowel"]).cuda()
        cons_weights = torch.FloatTensor(dl.weights["cons"]).cuda()

        grapheme_criterion = nn.CrossEntropyLoss(weight=grapheme_weights)
        vowel_criterion = nn.CrossEntropyLoss(weight=vowel_weights)
        cons_criterion = nn.CrossEntropyLoss(weight=cons_weights)
    else:
        grapheme_criterion = nn.CrossEntropyLoss()
        vowel_criterion = nn.CrossEntropyLoss()
        cons_criterion = nn.CrossEntropyLoss()

    if clr: 
        optimName = "clr"
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr_min, 
            momentum=.85,
            weight_decay=5e-4,
            nesterov=True)

        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            lr_min,
            lr_max,
            cycle_half_len*num_train_batches,
            cycle_half_len*num_train_batches,
            base_momentum=.85,
            max_momentum=.95
        )
    else: 
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=.2,
            patience=5
        )

    modelname = f'bengali-ai_densenet{model.net}-{weighted_classes_name}_'
                f'{num_epochs}_{optimizer_type}_{lr_min:.{5}}-{lr_max:.{5}}'


    ## Training
    for epoch in range(num_epochs): 
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*40)

        # change the lr_schedule from Cyclic to linear decrease for last few 
        # epochs
        if clr and epoch == cycle_len: 
            optimizer = optim.SGD(
                model.parameters(), 
                lr=lr_min, 
                nesterov=True, 
                momentum=.9) 
                        
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                num_train_batches, 
                lr_delta_adam)

        # always do a train and a validation phase
        for phase in ["train", "val"]: 
            grapheme_running_loss = 0.0
            vowel_running_loss = 0.0
            cons_running_loss = 0.0    
            grapheme_running_corrects = 0.0
            vowel_running_corrects = 0.0
            cons_running_corrects = 0.0 
           
            if phase == "train":
                model.train()
            else: 
                model.eval()

            t = dl.sizes[phase]//batch_size+1
            for batch, (inputs, labels) in tqdm(enumerate(dl[phase]), total=t):
                inputs, labels = inputs.to(device), labels.to(device) 
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    grapheme_out, vowel_out, cons_out = model(inputs)
                
                    _, grapheme_preds = torch.max(grapheme_out, 1)
                    _, vowel_preds = torch.max(vowel_out, 1)
                    _, cons_preds = torch.max(cons_out, 1)

                    grapheme_loss = grapheme_criterion(
                        grapheme_out,
                        labels[:,0])
                    vowel_loss = vowel_criterion(vowel_out, labels[:,1])
                    cons_loss = cons_criterion(cons_out, labels[:,2])
                    
                    if phase == "train":
                        (grapheme_loss+vowel_loss+cons_loss).backward()
                        optimizer.step()
                        if clr: 
                            lr_scheduler.step()
                

                grapheme_running_loss += grapheme_loss.item() * inputs.size(0)
                vowel_running_loss += vowel_loss.item() * inputs.size(0)
                cons_running_loss += cons_loss.item() * inputs.size(0)
                
                grapheme_corrects = torch.sum(grapheme_preds==labels[:,0].data)
                grapheme_running_corrects += grapheme_corrects
                
                vowel_corrects = torch.sum(vowel_preds==labels[:,1].data)
                vowel_running_corrects += vowel_corrects

                cons_corrects = torch.sum(cons_preds==labels[:,2].data)
                cons_running_corrects += cons_corrects

            running_losses = [
                grapheme_running_loss,
                vowel_running_loss,
                cons_running_loss,
            ]            

            if not clr and phase == "train":
                lr_scheduler.step(sum(running_losses))

        #end of epoch validation metrics 
        smpls = batch_size*(batch+1)

        epoch_loss = [l/smpls for l in running_losses]

        running_corrects = [
            grapheme_running_corrects,
            vowel_running_corrects,
            cons_running_corrects,
        ]
        epoch_acc = [l.item()/smpls for l in running_corrects]

        print(  f'Loss: {" ".join([f"{iv:{.4}}" for iv in epoch_loss])}'
                f'Acc: {" ".join([f"{iv:{.4}}" for iv in epoch_acc])} {" "*20}')

        #save model at the end of training
        torch.save(model, f"models/{starttime}/{modelname}.{epoch}")
        print("model saved")


if __name__ == "__main__": 
    train_model()
