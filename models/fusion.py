from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block


class Action_Recognition_Transformer(nn.Module):
    def __init__(self, device='cpu',  mocap_frames=600, acc_frames=150, num_joints=29, in_chans=3, acc_coords=3, acc_features=18, spatial_embed=32, sdepth=4,adepth=4,tdepth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, op_type='cls', embed_type='lin',fuse_acc_features=False,
                 drop_rate=0.05, attn_drop_rate=0.05, drop_path_rate=0.2,  norm_layer=None, num_classes=6):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            mocap_frames (int): input frame number for skeletal joints
            acc_frames (int): input num frames for acc sensor
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            acc_coords(int): number of coords in one acc reading from meditag sensor: (x,y,z)=3
            spatial_embed (int): spatial patch embedding dimension 
            sdepth (int): depth of spatial  transformer
            tdepth (int): depth of temporal transformer
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
            fuse_acc_features(bool): Wether to fuse acceleration feature into the acc feature or not!

        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) #Momil:embed_dim_ratio is spatial transformer's patch embed size
        temp_embed = spatial_embed*(num_joints)   #### temporal embed_dim is spatial embed*(num_jonits) - as one frame is of this dim! and v have acc_frames + sp_frames
        temp_frames = mocap_frames  #Input frames to the temporal transformer are frames from mocap sensor!
        acc_embed = temp_embed #Since both signals needs to be concatenated, their dm is similar
        self.op_type = op_type
        self.embed_type = embed_type
        
        print("-------------ACCELERATION-------------")
        print("Acc Frames: ",acc_frames)
        print("Acc embed dim: ",acc_embed)
        print("Acc depth: ",adepth)
        
        print('-----------SKELETON---------------')
        print("Temporal input tokens (Frames): ",mocap_frames)
        print("Spatial input tokens (Joints): ",num_joints)
        print("Spatial embed dim: ",spatial_embed)
        print("Temporal embed dim: ",temp_embed)
        print("Spatial depth: ",sdepth)
        print("Temporal depth: ",tdepth)

        print('-------------Regularization-----------')
        print("Drop Rate: ",drop_rate)
        print("Attn drop rate: ",attn_drop_rate) 
        print("Drop path rate: ",drop_path_rate)
        
        #Spatial patch and pos embeddings
        if embed_type=='lin':
            self.Spatial_patch_to_embedding = nn.Linear(in_chans, spatial_embed)#Linear patch embedding
        else:
            self.Spatial_patch_to_embedding = nn.Conv1d(in_chans, spatial_embed, 1, 1)#Conv patch embedding
        
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints+1, spatial_embed))
        self.spat_token = nn.Parameter(torch.zeros(1,1,spatial_embed))
        self.proj_up_clstoken = nn.Linear(mocap_frames*spatial_embed, num_joints*spatial_embed)
        self.sdepth = sdepth
        self.num_joints = num_joints
        self.joint_coords = in_chans

        #Temporal embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, temp_frames+1, temp_embed)) #additional pos embedding zero for class token
        self.temp_frames = mocap_frames

        #Acceleration patch and pos embeddings
        if embed_type=='lin':
            self.acc_coords_for_embedding = nn.Linear(acc_coords, acc_embed) #Linear patch embedding
        else:
            self.acc_coords_for_embedding = nn.Conv1d(acc_coords, acc_embed, 1, 1) #Conv patch embedding
        
        self.Acc_pos_embed = nn.Parameter(torch.zeros(1, acc_frames+1, acc_embed)) #1 location per frame - embed: 1xloc_embed from 1xloc_cords
        self.acc_token = nn.Parameter(torch.zeros(1,1,acc_embed))
        self.acc_frames = acc_frames
        self.acc_coords= acc_coords
        self.fuse_acc_features = fuse_acc_features
        self.acc_features = acc_features

        
        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]  #Stochastic depth decay rule
        adpr = [x.item() for x in torch.linspace(0, drop_path_rate, adepth)]  #Stochastic depth decay rule
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]  #Stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.Acceletaion_blocks = nn.ModuleList([
            Block(
                dim=acc_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i], norm_layer=norm_layer)
            for i in range(adepth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed)
        self.Acc_norm = norm_layer(acc_embed)
        self.Temporal_norm = norm_layer(temp_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        #Linear layer to extract features from the acc features signal
        self.acc_features_embed = nn.Linear(acc_features,acc_embed)

        #Classification head
        self.class_head = nn.Sequential(
            nn.LayerNorm(acc_embed+temp_embed),
            nn.Linear(acc_embed+temp_embed, num_classes)
        )

    def acc_forward_features(self,x):
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
        x = rearrange(x, 'b f p c  -> b f (p c)', ) # b x Fa x 3
        
        if self.embed_type == 'conv': 
            # If the embed_type is 'conv', reshape the input tensor to (B, F, C, P) for convolutional embedding
            x = rearrange(x, '(b f) p c  -> (b f) c p',b=b) # b x 3 x Fa  - Conv k liye channels first
            x = self.acc_coords_for_embedding(x) # B x c x p ->  B x Sa x p
            x = rearrange(x, '(b f) Sa p  -> (b f) p Sa', b=b)
        else: 
            # Else, perform linear embedding
            x = self.acc_coords_for_embedding(x) #all acceleration data points for the action = Fa | op: b x Fa x Sa

        # Create a class token for each frame
        class_token=torch.tile(self.acc_token,(b,1,1)) #(B,1,1) - 1 cls token for all frames

        # Concatenate the class token with the input tensor
        x = torch.cat((x,class_token),dim=1) 
        
        # Get the shape of the output tensor
        _,_,Sa = x.shape
        
        # Add positional embeddings to the input tensor
        x += self.Acc_pos_embed
        
        # Apply dropout to the input tensor
        x = self.pos_drop(x)
        
        # Iterate over all the blocks in the accelerometer transformer
        for blk in self.Acceletaion_blocks:
            x =blk(x)
        
        # Apply normalization to the output tensor
        x = self.Acc_norm(x)
        
        #Extract cls token
        cls_token = x[:,-1,:]
        if self.op_type=='cls':
            return cls_token
        else:
            # Reshape the output tensor to (B, F, Sa)
            x = x[:,:f,:]
            x = rearrange(x, 'b f Sa -> b Sa f')
            
            # Apply average pooling to the output tensor along the frame axis
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x Sa x 1
            
            # Reshape the output tensor to (B, Sa)
            x = torch.reshape(x, (b,Sa))
            return x #b x Sa

    def spatial_forward_features(self, x):
        """
        Forward pass for the spatial transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, F, P, C), where B is the batch size,
                             F is the number of frames, P is the number of joints, and C is the number of channels.
        
        Returns:
            torch.Tensor: Tuple containing the spatially encoded features (B, F, P*Se) and the class token (B, F*Se).
                          If op_type is 'cls', the class token is returned instead of the spatially encoded features.
        """
        
        # Get the batch size, number of frames, number of joints, and number of channels from the input tensor
        b, f, p, c = x.shape  # b is batch size, f is number of frames, p is number of joints, c is in_chan 3 
        
        # Reshape the input tensor to (B*F, P, C)
        x = rearrange(x, 'b f p c  -> (b f) p c', ) 
        
        if self.embed_type == 'conv':
            # If the embed_type is 'conv', reshape the input tensor to (B*F, C, P) for convolutional embedding
            x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
            x = self.Spatial_patch_to_embedding(x) # B x c x p ->  B x Se x p
            x = rearrange(x, '(b f) Se p  -> (b f) p Se', b=b)
        else: 
            # Else, perform linear embedding
            x = self.Spatial_patch_to_embedding(x) # B x p x c ->  B x p x Se
        
        # Create a class token for each frame
        class_token=torch.tile(self.spat_token,(b*f,1,1)) #(B,1,1) - 1 cls token for all frames
        
        # Concatenate the class token with the input tensor
        x = torch.cat((x,class_token),dim=1) # b x (p+1) x Se 
        
        # Add positional embeddings to the input tensor
        x += self.Spatial_pos_embed
        
        # Apply dropout to the input tensor
        x = self.pos_drop(x)
        
        # Iterate over all the blocks in the spatial transformer
        for blk in self.Spatial_blocks:
            x = blk(x)
        
        # Apply normalization to the output tensor
        x = self.Spatial_norm(x)
        
        #Extract cls token
        Se = x.shape[-1]
        cls_token = x[:,-1,:]
        cls_token = torch.reshape(cls_token, (b,f*Se))
        
        #Reshape input
        x = x[:,:p,:]
        x = rearrange(x, '(b f) p Se-> b f (p Se)', f=f)
        
        # Return the spatially encoded features and the class token
        if self.op_type=='cls':
            return cls_token
        else:
            return x, cls_token #cls token and encoded features returned

    def temporal_forward_features(self, x, cls_token):
        """
        Forward pass of the temporal transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, mocap_frames, num_joints*Se).
            cls_token (torch.Tensor): Class token tensor of shape (batch_size, num_joints*Se).
        
        Returns:
            torch.Tensor: If `self.op_type` is 'cls', returns the class token tensor of shape (batch_size, num_joints*Se).
                          Otherwise, returns the averaged features tensor of shape (batch_size, St).
        """
        
        # Extract batch size, mocap frames, and sequence length from the input tensor
        b,f,St = x.shape
        
        # Concatenate the input tensor with the class token along the sequence length dimension
        x = torch.cat((x,cls_token), dim=1) # B x mocap_frames +1 x temp_embed | temp_embed = num_joints*Se
        
        # Update batch size
        b  = x.shape[0]
        
        # Add positional embeddings to the input tensor
        x += self.Temporal_pos_embed
        
        # Apply dropout to the input tensor
        x = self.pos_drop(x)
        
        # Iterate over all the blocks in the temporal transformer
        for blk in self.Temporal_blocks:
            x = blk(x)

        # Apply normalization to the output tensor
        x = self.Temporal_norm(x)
        
        # Extract the class token from the output tensor
        if self.op_type=='cls':
            cls_token = x[:,-1,:]
            cls_token = cls_token.view(b, -1) # (Batch_size, temp_embed)
            return cls_token

        # If `self.op_type` is not 'cls', average the features over the mocap frames dimension
        else:
            x = x[:,:f,:]
            x = rearrange(x, 'b f St -> b St f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x St x 1
            x = torch.reshape(x, (b,St))
            return x #b x St


    def forward(self, inputs):
        """
        Forward pass of the fusion model.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, mocap_frames, num_joints, 3).
        
        Returns:
            torch.Tensor: Log softmax of the output tensor, shape (batch_size, St).
        """
        
        # Extract batch size, mocap frames, and number of joints from the input tensor
        batch_size, mocap_frames, num_joints, _ = inputs.shape
        
        # Extract skeletal signal from input
        skeletal_signal = inputs[:,:, :num_joints, :self.joint_coords] # B x Fs x num_joints x 3
        
        # Extract acceleration signal from input
        acc_features = inputs[:, 0, self.num_joints:self.num_joints+self.acc_features, 0 ] # B x 1 x acc_features x 1
        acc_signal = inputs[:, 0 , self.num_joints+self.acc_features:, :self.acc_coords] # B x 1 x Fa x 3
        acc_signal = torch.reshape(acc_signal, (batch_size,-1,1,self.acc_coords) ) # B x Fa x 1 x 3
        
        # Get skeletal features using spatial transformer
        skeletal_features, cls_token = self.spatial_forward_features(skeletal_signal) # B x Fs x (num_joints*Se)
        
        # Project up class token to match dimensions with temporal transformer
        cls_token = self.proj_up_clstoken(cls_token) # B x num_joints*Se
        cls_token = torch.unsqueeze(cls_token,dim=1) # B x 1 x num_joints*Se
        
        # Get temporal features using temporal transformer
        temporal_features = self.temporal_forward_features(skeletal_features,cls_token) # B x St
        
        # Get acceleration features using acceleration transformer
        acc_features = self.acc_forward_features(acc_signal) # B x St
        acc_features_embed = self.acc_features_embed(acc_features)
        if self.fuse_acc_features:
            acc_features += acc_features_embed # Add the features signal to acceleration signal
        
        # Concatenate skeletal and acceleration features along frame dimension
        concatenated_features = torch.cat((temporal_features, acc_features),dim=1) # B x (St + St)
        
        # Apply class head to the concatenated features
        output = self.class_head(concatenated_features)
        
        # Apply log softmax to the output tensor
        output = F.log_softmax(output,dim=1)
        
        return output

