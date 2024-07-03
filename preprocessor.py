import os
import pandas as pd
import re
import config as cfg


def pose2idlabel(data_path, labels_path, aug_path=None, start_i=0):
    """
    This function maps data for one activity to an id and label.
    
    Arguments:
    - data_path (list): list of paths to find specified data modalities
    - labels_path (string): path to the labels file
    - aug_path (string): path to the augmented data
    - start_i (int): int value to start id of samples at
    
    Output:
    - pose2id (dict): dictionary with path to sample id
    Test: pose2id = {id-10:{mocap: [path,[w1,w2,w3 ... w30]. meditag: [path,[w1,w2,w3 ... w30]},...}
    Train : pose2id = {id-10:{mocap: [path,w1], meditag: [path,w1] },...}
    - id2label (dict): dictionary with sample id to label
    Test: labels = {id=10: 0, id-11: 2, ...}
    Train: labels2id = {id-0: 0, id-1: 2, ...}
    - i (int): Sample ids start from 0 or given id
    """
    
    # Create empty dictionaries to store data
    pose2id = dict()  # path to sample id
    id2label = dict()  # sample id to label
    i = start_i  # Sample ids start from 0 or given id
    
    # Get paths to csv files
    mocap_path = data_path[0]
    acc_path = data_path[1]
    
    # If both paths exist
    if mocap_path and acc_path:
        
        # Get list of all files in the mocap_path directory
        samples = os.listdir(mocap_path)
        
        # Iterate over each file in the directory
        for smpl in samples:
            
            # Get the full path to the file
            mocap_smpl_path = os.path.join(mocap_path, smpl)
            
            # If the file ends with '.csv'
            if mocap_smpl_path.endswith('.csv'):
                
                # Get the segment id from the file name
                segment_id = int(re.findall(r'\d+', smpl)[0])
                
                # Create a new entry in the pose2id dictionary
                # with the sample id as the key and a dictionary as the value
                # The value dictionary contains the path to the mocap and acc files
                pose2id['id-' + str(i)] = {}
                pose2id['id-' + str(i)]['mocap'] = [mocap_smpl_path, segment_id]
                pose2id['id-' + str(i)]['acc'] = [acc_path, segment_id]
                
                # Create a new entry in the id2label dictionary
                # with the sample id as the key and the label as the value
                id2label['id-' + str(i)] = get_ncrc_labels(segment_id, labels_path)
                
                # Increment the sample id
                i += 1

    return pose2id, id2label, i


#Function: Label generator for activity recognition for NCRC dataset
#input : segment id
def get_ncrc_labels(segment_id,labels_path):
    lbl_map={2:0, #vital signs measurement
            3:1, #blood collection 
            4:2, #blood glucose measurement
            6:3, #indwelling drop retention and connection
            9:4, #oral care
            12:5} #diaper exchange and cleaning of area
    df=pd.read_csv(labels_path)
    df.drop('subject',axis=1,inplace=True)
    labels_map={}
    for _,row in df.iterrows():
        labels_map[row['segment_id']]=lbl_map[row['activity_id']]
    return labels_map[segment_id]


def preprocess():
    """
    This function preprocesses the data for the NCRC dataset.
    It reads the data from the specified paths and creates the necessary dictionaries
    to be used for training and testing.
    It also prints the number of samples in the training and test sets.
    """
    #Get pose dir to id dict, and id to label dict

    # NCRC paths
    # The training and testing paths for mocap and acc data
    mocap_train_path=cfg.file_paths['mocap_train_path']
    mocap_test_path=cfg.file_paths['mocap_test_path']
    acc_train_path=cfg.file_paths['acc_train_path']
    acc_test_path=cfg.file_paths['acc_test_path']

    # The paths to the training and testing labels
    tr_labels_path=cfg.file_paths['tr_labels_path']
    tst_labels_path=cfg.file_paths['tst_labels_path']

    # If augmentation is needed, specify the path here
    aug_path=None

    # Get the training pose2id, labels and starting id
    tr_pose2id,tr_labels,start_i = pose2idlabel([mocap_train_path,acc_train_path],tr_labels_path,aug_path)

    # Get the testing pose2id, labels and starting id
    pose2id,labels,_ = pose2idlabel([mocap_test_path,acc_test_path],tst_labels_path,aug_path,start_i)

    # Create the partition dictionary with the training and testing keys
    partition=dict()
    partition['train']=list(tr_pose2id.keys())
    partition['test']=list(pose2id.keys())

    # Print the number of training and testing samples
    print('--------------DATA SPLIT----------')
    print("Train Sample: ",len(tr_pose2id))
    print("Test Samples: ",len(pose2id))

    # Merge the training and testing pose2id and labels dictionaries
    pose2id.update(tr_pose2id)
    labels.update(tr_labels)

    # Print the partitions are made
    print("Partitions are  Made!" )

    # Return the pose2id, labels and partition dictionaries
    return pose2id,labels,partition


