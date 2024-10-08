from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block


class Action_Recognition_Transformer(nn.Module):
    # This class is the code of the Acceleration Only model. It takes in the acceleration data, processes it using a transformer, and outputs the class label.
    def __init__(self, device, acc_frames=150, num_joints=29, in_chans=3, acc_coords=3, acc_embed=32, 
                 acc_features=18, adepth=4, num_heads=8, mlp_ratio=2., qkv_bias=True,
                 qk_scale=None, op_type='cls', embed_type='lin', fuse_acc_features=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_classes=6):
        """
        Initialize the Acceleration Only Model.

        Args:
            acc_frames (int): input num frames for acc sensor
            num_joints (int, tuple): joints number
            acc_coords(int): number of coords in one acc reading from meditag sensor: (x,y,z)=3
            adepth (int): depth of acc transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            op_type(string): 'cls' or 'gap', output of temporal and acc encoder is cls token or global avg pool of encoded features.
            embed_type(string): convolutional 'conv' or linear 'lin'
            acc_features(int): number of features extracted from acc signal
            fuse_acc_features(bool): Wether to fuse acceleration feature into the acc feature or not!
            acc_coords (int) = 3(xyz) or 4(xyz, magnitude)
        """
        # Initialize the superclass
        super().__init__()

        # Set the norm_layer to nn.LayerNorm with epsilon=1e-6 if it's not already set
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 

        # Set the operation type and embed type
        self.op_type = op_type
        self.embed_type = embed_type

        # Set the number of joints and the number of coordinates per joint
        self.num_joints= num_joints
        self.joint_coords = in_chans

        # Print the model configuration
        print("ACCELERATION Only Model!")
        print("-------------ACCELERATION-------------")
        print("Acc Frames: ",acc_frames)
        print("Acc embed dim: ",acc_embed)
        print("Acc depth: ",adepth)

        print('-------------Regularization-----------')
        print("Drop Rate: ",drop_rate)
        print("Attn drop rate: ",attn_drop_rate) 
        print("Drop path rate: ",drop_path_rate)

        # Create the linear or convolutional layer to convert the coordinates to the embedding space
        if embed_type=='lin':
            # Create a linear layer to convert the coordinates to the embedding space
            self.acc_coords_to_embedding = nn.Linear(acc_coords, acc_embed) #Linear patch embedding
        else:
            # Create a convolutional layer to convert the coordinates to the embedding space
            self.acc_coords_to_embedding = nn.Conv1d(acc_coords, acc_embed, 1, 1) #Conv patch embedding
        
        # Create the positional embedding for the accelerometer frames
        self.acc_pos_embed = nn.Parameter(torch.zeros(1, acc_frames+1, acc_embed)) #1 location per frame - embed: 1xloc_embed from 1xloc_cords
        self.acc_token = nn.Parameter(torch.zeros(1,1,acc_embed))
        self.acc_frames = acc_frames
        self.adepth = adepth
        self.acc_features = acc_features
        self.fuse_acc_features = fuse_acc_features
        self.acc_coords = acc_coords

        # Create the list of blocks for the accelerometer transformer
        adpr = [x.item() for x in torch.linspace(0, drop_path_rate, adepth)]  #Stochastic depth decay rule

        self.acceleration_blocks = nn.ModuleList([
            # Create a block for the accelerometer transformer
            Block(
                dim=acc_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i], norm_layer=norm_layer)
            for i in range(adepth)])

        # Create the normalization layer for the accelerometer embeddings
        self.acc_norm = norm_layer(acc_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Create the linear layer to extract features from the acceleration signal
        self.acc_features_embed = nn.Linear(acc_features,acc_embed)

        # Create the classification head for outputting the class label
        self.class_head = nn.Sequential(
            # Create a layer normalization layer
            nn.LayerNorm(acc_embed),
            # Create a linear layer to extract features from the acceleration signal
            nn.Linear(acc_embed, num_classes)
        )

    def acc_forward_features(self, x):
        """
        Forward pass for the accelerometer transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, F, P, C), where B is the batch size,
                             F is the number of frames, P is the number of points, and C is the number of channels.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, Sa), where Sa is the number of accelerometer embeddings.
                          If op_type is 'cls', the output tensor is of shape (B, C), where C is the number of classes.
        """
        
        # Get the batch size, number of frames, number of points, and number of channels from the input tensor
        b, f, p, c = x.shape  # b is batch size, f is number of frames, c is values per reading 3, p is reading per frames 1, B x Fa X 1 x 3
        
        # Reshape the input tensor to (B, F, P*C)
        x = rearrange(x, 'b f p c  -> b f (p c)', ) 
        
        # If the embed_type is 'conv', reshape the input tensor to (B, F, C, P) for convolutional embedding
        if self.embed_type == 'conv':
            x = rearrange(x, '(b f) p c  -> (b f) c p', b=b)  # b x 3 x Fa  - Conv k liye channels first
            x = self.acc_coords_to_embedding(x)  # B x c x p ->  B x Sa x p
            x = rearrange(x, '(b f) Sa p  -> (b f) p Sa', b=b)
        else: 
            # Else, perform linear embedding
            x = self.acc_coords_to_embedding(x)  # all acceleration data points for the action = Fa | op: b x Fa x Sa
        
        # Create a class token for each frame
        class_token = torch.tile(self.acc_token, (b, 1, 1))  # (B, 1, 1) - 1 cls token for all frames
        
        # Concatenate the class token with the input tensor
        x = torch.cat((x, class_token), dim=1) 
        
        # Get the shape of the output tensor
        _, _, Sa = x.shape
        
        # Add positional embeddings to the input tensor
        x += self.acc_pos_embed
        
        # Apply dropout to the input tensor
        x = self.pos_drop(x)
        
        # Iterate over all the blocks in the accelerometer transformer
        for _, blk in enumerate(self.acceleration_blocks):
            x = blk(x)
        
        # Apply normalization to the output tensor
        x = self.acc_norm(x)
        
        # Extract the class token from the output tensor
        cls_token = x[:, -1, :]
        
        # If the op_type is 'cls', return the class token
        if self.op_type == 'cls':
            return cls_token
        else:
            # Else, reshape the output tensor to (B, Sa, F), pool it along the frame dimension, and reshape it to (B, Sa)
            x = x[:, :f, :]
            x = rearrange(x, 'b f Sa -> b Sa f')
            x = F.avg_pool1d(x, x.shape[-1], stride=x.shape[-1])  # b x Sa x 1
            x = torch.reshape(x, (b, Sa))
            
            return x  # b x Sa

    def forward(self, inputs):
        #Input: B x MOCAP_FRAMES X  119 x 3
        b,_,_,c = inputs.shape

        #Extract skeletal signal from input
        # x = inputs[:,:, :self.num_joints, :self.joint_coords] #B x Fs x num_joints x 3

        #Extract acc signal from input
        sxf = inputs[:, 0, self.num_joints:self.num_joints+self.acc_features, 0 ] #B x 1 x acc_features x 1
        sx = inputs[:, 0 , self.num_joints+self.acc_features:, :self.acc_coords] #B x 1 x Fa x 3
        sx = torch.reshape(sx, (b,-1,1,self.acc_coords) ) #B x Fa x 1 x 3


        #Get acceleration features
        sx = self.acc_forward_features(sx)
        #print("Input to ACC Transformer: ",sx) #in: F x Fa x 3 x 1,  op: B x St
        sxf = self.acc_features_embed(sxf)
        if self.fuse_acc_features:
            sx+= sxf #Add the features signal to acceleration signal

        #Concat features along frame dimension
        sx = self.class_head(sx)

        return F.log_softmax(sx,dim=1)
