import pandas as pd
import numpy as np
import torch
import random
import config as cfg
import scipy.stats as s
pd.options.mode.chained_assignment = None  # default='warn'



MOCAP_SEGMENT = cfg.data_params['MOCAP_SEGMENT']
ACC_SEGMENT = cfg.data_params['ACC_SEGMENT']



#CREATE PYTORCH DATASET
'''
Input Args:
data = ncrc or ntu
num_frames = mocap and nturgb+d frame count!
acc_frames = frames from acc sensor per action
'''

class Poses3d_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, list_IDs, labels, pose2id,  **kwargs): 
        
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.pose2id = pose2id
        self.data=data
        
        if self.data=='ncrc':
            self.partition = kwargs.get('partition', None)
            self.normalize = kwargs.get('normalize', None)
            self.meditag = np.array([])
            self.meditag_segment_id = None
            self.mocap = np.array([])
            self.mocap_segment_id = None
            self.mocap_frames = kwargs.get('mocap_frames', None)
            self.acc_frames = kwargs.get('acc_frames',None)

        
    #Function to compute magnitude of a signal
    def magnitude(self,data):
        data = data.numpy()
        data[:,0] = data[:,0]**2
        data[:,1] = data[:,1]**2
        data[:,2] = data[:,2]**2
        return np.sqrt(data[:,0]+data[:,1]+data[:,2]).reshape(data.shape[0],1)

    #Function to select frames
    def frame_selection(self, data_sample, num_frames, skip=2):
        """
        This function selects frames from the given data_sample based on the specified number of frames.
        The function first checks if the number of frames in the data_sample is greater than the desired number of frames.
        If it is, it selects every second frame using the skip parameter.
        If the number of frames in the selected data_sample is still greater than the desired number of frames,
        it selects a random subset of frames until the desired number of frames is reached.
        If the number of frames in the data_sample is less than the desired number of frames,
        it adds additional frames by repeating the existing frames or by repeating the last frame until the desired number of frames is reached.
        
        Args:
            data_sample (numpy array): The array containing the data samples.
            num_frames (int): The desired number of frames.
            skip (int, optional): The number of frames to skip between each selected frame. Defaults to 2.
        
        Returns:
            numpy array: The selected frames with the desired number of frames.
        """
        
        # Check if the number of frames in the data_sample is greater than the desired number of frames
        if (data_sample.shape[0]) > num_frames:
            # Select every second frame using the skip parameter
            data_sample = np.array(data_sample[::skip, :, :])
            
            # If the number of frames in the selected data_sample is still greater than the desired number of frames,
            # select a random subset of frames until the desired number of frames is reached
            if data_sample.shape[0] > num_frames:
                # Calculate the number of frames to select randomly
                diff = num_frames - data_sample.shape[0]
                
                # Select diff random frames
                if diff <= data_sample.shape[0]:  # If diff is less than or equal to the number of frames in data_sample
                    if diff < data_sample.shape[0]:
                        # Select random indices from 0 to data_sample.shape[0]-1 except the last index
                        sampled_frames = random.sample(range(0, data_sample.shape[0] - 1), diff)
                    elif diff == data_sample.shape[0]:
                        # If diff is equal to the number of frames in data_sample, select all indices
                        sampled_frames = np.arange(data_sample.shape[0])
                    
                    # Insert the selected frames at their respective indices in the data_sample array
                    for f in sampled_frames:
                        data_sample = np.insert(data_sample, f, data_sample[f, :, :], axis=0)
                
                # If diff is greater than the number of frames in data_sample, repeat every frame twice
                elif diff > data_sample.shape[0]:
                    # Select all indices
                    sampled_frames = np.arange(data_sample.shape[0])
                    
                    # Insert the selected frames at their respective indices in the data_sample array
                    for f in sampled_frames:
                        data_sample = np.insert(data_sample, f, data_sample[f, :, :], axis=0)
                    
                    # Calculate the remaining number of frames to be added
                    rem_diff = num_frames - data_sample.shape[0]
                    
                    # Repeat the last frame pose for rem_diff times
                    last_frame = data_sample[-1, :, :]
                    tiled = np.tile(last_frame, (rem_diff, 1, 1))
                    data_sample = np.append(data_sample, tiled, axis=0)
        
        # If the number of frames in the data_sample is less than the desired number of frames,
        # add additional frames by repeating the existing frames or by repeating the last frame until the desired number of frames is reached
        if data_sample.shape[0] < num_frames:
            # Calculate the difference between the desired number of frames and the number of frames in the data_sample
            diff = num_frames - data_sample.shape[0]
            
            # Select every frame twice and repeat the last frame to complete the desired number of frames
            sampled_frames = np.arange(data_sample.shape[0])
            
            # Insert the selected frames at their respective indices in the data_sample array
            for f in sampled_frames:
                data_sample = np.insert(data_sample, f, data_sample[f, :, :], axis=0)
            
            # If still less than the desired number of frames, repeat the last frame pose until the desired number of frames is reached
            if data_sample.shape[0] < num_frames:
                rem_diff = num_frames - data_sample.shape[0]
                last_frame = data_sample[-1, :, :]
                tiled = np.tile(last_frame, (rem_diff, 1, 1))
                data_sample = np.append(data_sample, tiled, axis=0)
        
        return data_sample


    #Read segment - Function to read actiona nd convert it to fx29x3
    #Input: segment csv file path, window id
    #Output: Array: fx29x3
    def read_mocap_segment(self, path):
        """
        Reads a segment csv file containing mocap data and converts it to an array with shape (frames, 29, 3).
        The csv file is expected to have the following columns:
        - 'time_elapsed': time elapsed in seconds
        - 'segment_id': id of the segment
        - 'x1', 'y1', 'z1', ..., 'x29', 'y29', 'z29': coordinates of 29 body joints
        The function performs the following steps:
        1. Reads the csv file into a pandas DataFrame.
        2. Drops the 'time_elapsed' and 'segment_id' columns from the DataFrame.
        3. Interpolates missing values in the DataFrame using linear interpolation.
        4. Fills missing values at the beginning and end of each row using the last observed value.
        5. Reshapes the DataFrame into an array with shape (frames, 29, 3), where each row represents the coordinates of a body joint at a specific frame.
        6. Optionally, selects frames from the array using the `frame_selection` method.
        7. Optionally, normalizes the array using the `normalize_data` method.
        8. Returns the array with shape (frames, 29, 3).
        """
        # Read the csv file into a pandas DataFrame
        df = pd.read_csv(path)

        # Drop the 'time_elapsed' and 'segment_id' columns from the DataFrame
        df.drop('time_elapsed', axis=1, inplace=True)
        df.drop('segment_id', axis=1, inplace=True)

        # Interpolate missing values in the DataFrame using linear interpolation
        df.interpolate(method='linear', axis=0, inplace=True)

        # Fill missing values at the beginning and end of each row using the last observed value
        df.fillna(method='bfill', axis=0, inplace=True)
        df.fillna(method='ffill', axis=0, inplace=True)
        df.fillna(value=0, axis=0, inplace=True)

        # Reshape the DataFrame into an array with shape (frames, 29, 3)
        data = df.to_numpy()
        frames = data.shape[0]
        data = np.reshape(data, (frames, 29, 3))

        # Optionally, select frames from the array using the `frame_selection` method
        data = self.frame_selection(data, self.mocap_frames, skip=10)

        # Optionally, normalize the array using the `normalize_data` method
        if self.normalize:
            data = self.normalize_data(data, np.mean(data), np.std(data))

        # Return the array with shape (frames, 29, 3)
        return data  # 600 x 29 x 3


    #Read segment - Function to read acceleration data and convert it to 120 x 3 [x,y,z]
    #Input: segment csv file path, window id
    #Output: Array: 20x3
    def read_acc_segment(self,path,segment_id):
        # Read the csv file into a pandas DataFrame
        df = pd.read_csv(path)

        # Drop the 'time_elapsed' column from the DataFrame
        df.drop('time_elapsed',axis=1,inplace=True)

        # Interpolate missing values in the DataFrame using linear interpolation
        df.interpolate(method='linear',axis=0,inplace=True)

        # Fill missing values at the beginning and end of each row using the last observed value
        df.fillna(method='bfill',axis=0,inplace=True)

        # Extract the rows corresponding to the specified segment_id
        df_segment = df.loc[df['segment_id']==segment_id,:].copy()

        # Drop the 'segment_id' column from the DataFrame
        df_segment.drop('segment_id',axis=1,inplace=True)

        # Convert the DataFrame to a numpy array
        data=df_segment.to_numpy()

        # If the extracted data has zero rows, it means the segment is missing
        if data.shape[0]==0:
            # Extract segments 219 and 685, which are similar segments performed by the same subject
            segment219 =  df.loc[df['segment_id'] == 219,:]
            segment219.drop('segment_id',axis=1,inplace=True)
            data219 = segment219.to_numpy()

            segment685 =  df.loc[df['segment_id'] == 685,:]
            segment685.drop('segment_id',axis=1,inplace=True)
            data685 = segment685.to_numpy()

            # Calculate the minimum number of samples between the two segments
            min_samples = min(data685.shape[0], data219.shape[0])

            # Take the first min_samples samples from both segments
            data219 = data219[:min_samples,:]
            data685 = data685[:min_samples,:]

            # Calculate the mean of the two segments
            data = np.mean([data219,data685], axis=0)

        # If the extracted data has fewer samples than the desired number of frames, pad it with the last observed value
        if (data.shape[0]<self.acc_frames):
            last_loc = data[-1,:]
            diff = self.acc_frames - data.shape[0]
            tiled = np.tile(last_loc,(diff,1))
            data = np.append(data,tiled,axis=0)
        # If the extracted data has more samples than the desired number of frames, take only the first self.acc_frames samples
        elif data.shape[0]>self.acc_frames:
            data=data[:self.acc_frames,:]

        # If normalization is enabled, normalize the data
        if self.normalize:
            data = self.normalize_data(data,np.mean(data),np.std(data))

        # Return the numpy array with shape (self.acc_frames, 3) representing the acceleration data for the specified segment
        return data #self.acc_frames x 3

    def extract_acc_features(self,data):
        #[mean(xyz), std(xyz), Max(xyz), Min(xyz), Kurtosis(xyz), Skewness(xyz)]
        data = data.numpy()
        acc_features=torch.zeros((18,4))
        #Mean
        acc_features[0,0] = np.mean(data[:,0])
        acc_features[1,0] = np.mean(data[:,1])
        acc_features[2,0] = np.mean(data[:,2])

        #Std
        acc_features[3,0] = np.std(data[:,0])
        acc_features[4,0] = np.std(data[:,1])
        acc_features[5,0] = np.std(data[:,2])

        #Max
        acc_features[6,0] = np.max(data[:,0])
        acc_features[7,0] = np.max(data[:,1])
        acc_features[8,0] = np.max(data[:,2])

        #Min
        acc_features[9,0] = np.min(data[:,0])
        acc_features[10,0] = np.min(data[:,1])
        acc_features[11,0] = np.min(data[:,2])

        #Kurtosis
        acc_features[12,0] = s.kurtosis(data[:,0], fisher=False)
        acc_features[13,0] = s.kurtosis(data[:,1], fisher=False)
        acc_features[14,0] = s.kurtosis(data[:,2], fisher=False)

        #Skewness
        acc_features[15,0] = s.skew(data[:,0])
        acc_features[16,0] = s.skew(data[:,1])
        acc_features[17,0] = s.skew(data[:,2])

        return acc_features

    def normalize_data(self,data,mean,std):
        return (data-mean) / std

  
    #Function to get poses for F frames/ one sample, given sample id 
    def get_pose_data(self,id):
        """
        This function retrieves the pose data for a given sample id.
        
        Parameters:
        - id (int): The sample id for which the pose data is to be retrieved.
        
        Returns:
        - data_sample (torch.Tensor): A tensor containing the pose data for the given sample id.
        """
        
        # Check if the data is 'ncrc'
        if self.data =='ncrc':
            
            # Get the information about the paths to the action sample
            segment_info = self.pose2id[id]
            
            # Get the path to one windowed sample of mocap data - [path, int id]
            mocap_info = segment_info['mocap']
            
            # Get the path to one windowed sample of acc data - [path, int id]
            acc_info = segment_info['acc']
            
            # Extract the segment ids
            segment_id = mocap_info.pop()
            segment_id = acc_info.pop()
            
            # Read the mocap segment and convert it to a tensor
            mocap_sig = torch.tensor( self.read_mocap_segment( mocap_info[0] ) )
            
            # Read the acc segment and convert it to a tensor
            acc_sig = torch.tensor( self.read_acc_segment( acc_info[0] , segment_id ) )
            
            # Extract the acceleration features
            acc_features = self.extract_acc_features(acc_sig)
            
            # Add the magnitude signal to the acceleration
            acc_mag = torch.from_numpy(self.magnitude(acc_sig))
            acc_sig = torch.cat((acc_sig,acc_mag), dim=1)
            
            # Concatenate the features and acc data
            acc_data = torch.cat((acc_features,acc_sig),dim=0)
            
            # Create a tensor to hold the extended acc data
            acc_ext = torch.zeros((self.mocap_frames, self.acc_frames + len(acc_features), 4)) #len(acc_feature) inplace of ACC_FEATURES
            
            # Copy the acc data to the acc_ext tensor
            acc_ext[0,:,:] = acc_data
            
            # Create a tensor to hold the extended mocap data
            mocap_ext = torch.zeros((self.mocap_frames,29,4))
            
            # Copy the mocap data to the mocap_ext tensor
            mocap_ext[:,:,:3] = mocap_sig
            
            # Concatenate the extended mocap and acc data
            data_sample = torch.cat((mocap_ext,acc_ext),dim=1)
            
            # Return the data_sample tensor
            return data_sample


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            # Load data and get label
            data=self.get_pose_data(ID)
            if isinstance(data,np.ndarray):
                X = torch.from_numpy(data)
            else:
                X = data
            y = self.labels[ID]
            return X, y
