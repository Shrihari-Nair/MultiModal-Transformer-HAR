import torch
import numpy as np
from Make_Dataset import Poses3d_Dataset
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActRecogTransformer
from Plotter.visualize import get_plot
import pickle
from asam import ASAM, SAM
from timm.loss import LabelSmoothingCrossEntropy
import os

exp = 'myexp-1' #Assign an experiment id

if not os.path.exists('exps/'+exp+'/'):
    os.makedirs('exps/'+exp+'/')
PATH='exps/'+exp+'/'

print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 3}
max_epochs = 250

pose2id, labels, partition = PreProcessing_ncrc.preprocess()