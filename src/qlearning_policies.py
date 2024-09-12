import signatory
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigQFunction(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, 
                 initial_basepoint=None, initial_bias=0.1): 
        super().__init__()
        self.sig_depth = sig_depth
        self.initial_basepoint = (
            torch.tensor(initial_basepoint, requires_grad=False,dtype=torch.float).unsqueeze(0)
            if initial_basepoint not in (None, True) else initial_basepoint
        )
        self.sig_channels = signatory.signature_channels(channels=in_channels,
                                                         depth=sig_depth)
        self.linear = torch.nn.Linear(self.sig_channels, out_dimension, bias=True)
        self.linear.bias.data.fill_(initial_bias)
        nn.init.xavier_uniform_(self.linear.weight)
        

    def forward(self, signature):
        """ 
        The signature of a path fed through a single linear layer.
        :signature: is a two dimensional tensor of shape (batch, self.sig_channels). 
        Returns a two dimensional tensor of shape (batch, out_dimension).      
        """
        return self.linear(signature)
    
    
    def compute_signature(self, path, with_basepoint=True):
        """
        This functions returns the signature of a given path. If with_basepoint=True,
        the value self.initial_basepoint is used as basepoint for the signature 
        calculation.
        
        Args:
            - path: a three dimensional tensor of shape (batch, length, in_channels)
            - with_basepoint: boolean
        """
        if with_basepoint:
            if self.initial_basepoint==None:
                return signatory.signature(path, depth=self.sig_depth,
                                           basepoint=path[:,0,:])
            else:
                return signatory.signature(path, depth=self.sig_depth,
                                           basepoint=self.initial_basepoint)
        else:
            return signatory.signature(path, depth=self.sig_depth,
                                       basepoint=False)
            
    def extend_signature(self, new_path, basepoint, signature):
        """
        This function updates a given signature with a new stream where basepoint
        is the last value of the old path from which :signature: was computed
        
        Args:
            - new_path: a three dimensional tensor of shape (batch, length, in_channels)
            - basepoint: a three dimentional tensor of shape (batch, 1, in_channels)
            - signature: a two dimensional tensor of shape (batch, self.sig_channels)
        """
        return signatory.signature(new_path, depth=self.sig_depth,
                                   basepoint=basepoint, initial=signature)
    
    # TODO: adapt training algo to use shorten_signature method if history window is fixed
    def shorten_signature(self, old_path, signature_full_path):
        """
        This function returns the signature of a shortened path with some old path section
        at its beginning removed. old_path must be a path section, starting at time 0 and 
        running until some time t, of the path from which signature was computed in the first place.
        
        Args:
            - old_path: a three dimensional tensor of shape (batch, length, in_channels), must
              be the some initial section of the path from which signature was computed
            - signature: a two dimensional tensor of shape (batch, self.sig_channels)
        """
        old_path_reverse = torch.flipud(old_path.squeeze(0)).unsqueeze(0)
        if self.initital_basepoint==None:
            sig_old_path_reverse = signatory.signature(old_path_reverse, 
                                                     depth=self.sig_depth,
                                                     basepoint=old_path[:,0,:])
        else:
            sig_old_path_reverse = signatory.signature(old_path_reverse, 
                                                     depth=self.sig_depth,
                                                     basepoint=self.initial_basepoint)
        return signatory.signature_combine(sig_old_path_reverse, signature_full_path,
                                           input_channels=self.sig_channels, 
                                           depth=self.sig_depth)
                

class RNNQFunction(nn.Module):
    def __init__(self, in_channels, out_dimension, layers=1, **kwargs):
        super().__init__(**kwargs)
        
        self.rnn = nn.RNN(in_channels, 32, layers, nonlinearity="relu", batch_first=True)
        # specify hidden layers when initialzed
        self.fc1 = nn.Linear(32, out_dimension)
        
    def forward(self, seq):
        out, _ = self.rnn(seq)
        out = out[:, -1, :]
        out = self.fc1(out)
        return out
    

class DSTQFunction(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth):
        super().__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(in_channels,),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=False)
        self.signature = signatory.Signature(depth=sig_depth)
        sig_channels = signatory.signature_channels(channels=in_channels + 2, 
                                                    # + 2 for current time and position
                                                    depth=sig_depth)

        self.linear1 = nn.Linear(sig_channels + 2, 64)
        self.linear2 = nn.Linear(64, out_dimension)

    def forward(self, seq):
        x = self.augment(seq)
        x = self.signature(x, basepoint=True)
        x = torch.cat([x, seq[:, :, -1]], dim=-1)
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)
    


