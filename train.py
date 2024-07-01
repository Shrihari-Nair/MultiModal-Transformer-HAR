import torch
import numpy as np
from create_dataset import Poses3d_Dataset
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActRecogTransformer
from Plotter.visualize import get_plot
import pickle
from asam import ASAM, SAM
from timm.loss import LabelSmoothingCrossEntropy
import os

exp = 'myexp-1'

if not os.path.exists('exps/'+exp+'/'):
    os.makedirs('exps/'+exp+'/')
PATH='exps/'+exp+'/'

print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# cuDNN will perform a set of heuristics to find the most optimal set of algorithms for your specific 
# hardware configuration and the given input size/dimension.
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 3}
max_epochs = 250

pose2id, labels, partition = PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")
mocap_frames = 600
acc_frames = 150

training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=labels, 
                               pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=False)
training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

