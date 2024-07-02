import torch
from create_dataset import Poses3d_Dataset
import preprocessor
from models.cross_view_fusion import Action_Recognition_Transformer
from plotter.visualizer import get_plot
from asam import ASAM
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

pose2id, labels, partition = preprocessor.preprocess()

print("Creating Data Generators...")
mocap_frames = 600
acc_frames = 150

training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=labels, 
                               pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=False)
training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3 

validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels, pose2id=pose2id, 
                                 mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=False)
validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

print("Initiating Model...")
model = Action_Recognition_Transformer(device)
model = model.to(device)

# Define Loss Function and Optimizer
lr=0.0025
wt_decay=5e-4
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wt_decay)

#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

#Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, max_epochs)
print("Using cosine")

#TRAINING AND VALIDATING
epoch_loss_train=[]
epoch_loss_val=[]
epoch_acc_train=[]
epoch_acc_val=[]

#Label smoothing
smoothing=0.1
criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
print("Loss: LSC ",smoothing)

best_accuracy = 0.


print("Begin Training....")
for epoch in range(max_epochs):
    # Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    for inputs, targets in training_generator:
        inputs = inputs.to(device) 
        #print("Input batch: ",inputs)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Ascent Step
        #print("labels: ",targets)
        predictions = model(inputs.float())
        #print("predictions: ",torch.argmax(predictions, 1) )
        batch_loss = criterion(predictions, targets)
        batch_loss.mean().backward()
        minimizer.ascent_step()

        # Descent Step
        criterion(model(inputs.float()), targets).mean().backward()
        minimizer.descent_step()

        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
    loss /= cnt
    accuracy *= 100. / cnt
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    #scheduler.step()

    #accuracy,loss = validation(model,validation_generator)
    #Test
    model.eval()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    model=model.to(device)
    with torch.no_grad():
        for inputs, targets in validation_generator:

            b = inputs.shape[0]
            inputs = inputs.to(device)
            #print("Validation input: ",inputs)
            targets = targets.to(device)
            
            predictions = model(inputs.float())
            
            with torch.no_grad():
                loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt
        
    
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(),PATH+exp+'_best_ckpt.pt')
            print("Check point "+PATH+exp+'_best_ckpt.pt'+ ' Saved!')

    print(f"Epoch: {epoch},Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")


    epoch_loss_val.append(loss)
    epoch_acc_val.append(accuracy)


print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')