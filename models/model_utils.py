import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Initialize the Mlp class.
        
        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to None.
                If None, then it defaults to the value of in_features.
            out_features (int, optional): Number of output features. Defaults to None.
                If None, then it defaults to the value of in_features.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Dropout rate. Defaults to 0.
        """
        # Call the parent class's (nn.Module) constructor
        super().__init__()
        
        # If out_features is None, then set it to the value of in_features
        out_features = out_features or in_features
        
        # If hidden_features is None, then set it to the value of in_features
        hidden_features = hidden_features or in_features
        
        # Create a linear layer with in_features as the number of input features
        # and hidden_features as the number of output features
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # Create an instance of the activation layer specified by act_layer
        self.act = act_layer()
        
        # Create a linear layer with hidden_features as the number of input features
        # and out_features as the number of output features
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Create a dropout layer with drop as the dropout rate
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the Mlp module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        
        # Apply the first linear layer, which performs a linear transformation of the input
        # The output shape is (batch_size, hidden_features)
        x = self.fc1(x)
        
        # Apply the activation function to the output of the linear layer
        # The output shape remains the same
        x = self.act(x)
        
        # Apply dropout to the output of the activation function
        # The output shape remains the same
        x = self.drop(x)
        
        # Apply the second linear layer, which performs a linear transformation of the output
        # The output shape is (batch_size, out_features)
        x = self.fc2(x)
        
        # Apply dropout to the output of the second linear layer
        # The output shape remains the same
        x = self.drop(x)
        
        # Return the final output tensor
        return x

#Attention computation
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Initialize the Attention module.
        
        Args:
            dim (int): Dimension of the input features.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): If True, adds a learnable bias to the query, key, and value. Defaults to False.
            qk_scale (float, optional): Scale factor for the query and key matrices. Defaults to None.
            attn_drop (float, optional): Dropout rate for the attention weights. Defaults to 0.
            proj_drop (float, optional): Dropout rate for the output features. Defaults to 0.
        """
        super().__init__()  # Call the constructor of the superclass (nn.Module)
        
        # Set the number of attention heads
        self.num_heads = num_heads
        
        # Calculate the dimension of each head
        head_dim = dim // num_heads
        
        # Set the scale factor for the query and key matrices
        # NOTE: The original implementation had a wrong scale factor, so we allow the user to set it manually
        self.scale = qk_scale or head_dim ** -0.5
        
        # Initialize the linear layer for query, key, and value
        # The input dimension is dim, and the output dimension is 3 times dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Initialize the dropout layer for the attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Initialize the linear layer for the output features
        # The input dimension is dim, and the output dimension is dim
        self.proj = nn.Linear(dim, dim)
        
        # Initialize the dropout layer for the output features
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Reshape the input tensor x to be (Batch, Num of tokens, embed dim)
        B, N, C = x.shape
        
        # Compute the query, key, and value matrices using the linear layer
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # Split the query, key, and value matrices into separate tensors
        # This is done to make the code compatible with PyTorch Script (torchscript does not support tuples of tensors)
        # Instead, we use tensor indexing to access the elements of the tensor
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute the attention weights by taking the dot product of the query and key matrices
        # The attention weights are then scaled by the scale factor
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply the softmax function to the attention weights to obtain the attention probabilities
        attn = attn.softmax(dim=-1)
        
        # Apply the dropout layer to the attention weights
        attn = self.attn_drop(attn)
        
        # Compute the output features by taking the dot product of the attention weights and value matrices
        # The output features are then reshaped to be (Batch, Num of tokens, embed dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Apply the linear layer to the output features
        x = self.proj(x)
        
        # Apply the dropout layer to the output features
        x = self.proj_drop(x)
        
        # Return the output features
        return x


#Cross View Attention computation
class CVAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.Wq = nn.Linear(dim, dim , bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim , bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xkv):
        B, N, C = xq.shape  #Batch x Num of tokens x embed dim
        B, n, C = xkv.shape
        q = self.Wq(xq).reshape( N, -1) # B,  self.num_heads, C//self.num_heads)
        k = self.Wk(xkv).reshape( -1, n) # B,  self.num_heads, C//self.num_heads, n)
        v = self.Wv(xkv).reshape( -1, n) # B,  self.num_heads, C//self.num_heads, n)

        #Compute attn weights
        #q - B,N,C
        #k,v - B,n,C
        attn = torch.matmul(q,k) * self.scale #Nxn
        #attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = torch.matmul(attn,v.T).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#Drop path
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#Transformer Block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, blocktype=None):
        """
        Initialize the Transformer Block.

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of the hidden dimension to the input dimension in the MLP.
                Defaults to 4.
            qkv_bias (bool, optional): If True, adds a learnable bias to the query, key, and value.
                Defaults to False.
            qk_scale (float, optional): Scale factor for the query and key matrices.
                Defaults to None.
            drop (float, optional): Dropout rate for the input features.
                Defaults to 0.
            attn_drop (float, optional): Dropout rate for the attention weights.
                Defaults to 0.
            drop_path (float, optional): Stochastic depth rate.
                Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            blocktype (str, optional): Type of block. Defaults to None.
        """
        super().__init__()
        
        # Initialize the normalization layers
        self.norm1 = norm_layer(dim)  # Normalization layer after the self-attention layer
        self.norm2 = norm_layer(dim)  # Normalization layer after the MLP
        
        # Initialize the self-attention layer
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # Initialize the drop path layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Initialize the cross-view attention layer
        self.cross_attn=CVAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop)
        
        # Calculate the hidden dimension for the MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # Initialize the MLP
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Set the block type
        self.blocktype=blocktype

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
                If the blocktype is 'Sensor', it returns a tuple of two tensors:
                cv_signal (torch.Tensor): The tensor detached and cloned from x.
                x (torch.Tensor): The input tensor after passing through the block.
        """
        # Initialize cv_signal as None
        cv_signal = None

        # Apply self-attention and add the result to x
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # If the blocktype is 'Sensor', detach and clone x and store it in cv_signal
        if self.blocktype == 'Sensor':
            cv_signal = x.detach().clone()

        # Apply the MLP and add the result to x
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # If the blocktype is 'Sensor', return cv_signal and x as a tuple
        if self.blocktype == 'Sensor':
            return cv_signal, x
        else:
            # Otherwise, return x alone
            return x

    def cross_forward(self,xq,xkv):
        """
        Forward pass of the block for cross view attention.

        Args:
            xq (torch.Tensor): Input tensor from the query view of shape (batch_size, channels, height, width).
            xkv (torch.Tensor): Input tensor from the key and value view of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        # Pass xkv through the normalization layer 1
        xkv=self.norm1(xkv)

        # Pass xq through the normalization layer 1
        xq=self.norm1(xq)

        # Apply self-attention to xq
        xq = xq + self.drop_path(self.attn(self.norm1(xq)))

        # Apply cross view attention to xq and xkv
        x = xq + self.drop_path(self.cross_attn(xq,xkv))

        # Apply the MLP to x
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # Return the output tensor x
        return x

