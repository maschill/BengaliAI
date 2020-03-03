from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.train import train_model

if __name__=="__main__":
    # training cyclic lr 2 hrs
    # train_model(num_epochs=35)
    
    #training constant lr
    # train_model(num_epochs=25, clr=False)
    train_model(num_epochs=35, clr=False, weighted_classes=True)
    