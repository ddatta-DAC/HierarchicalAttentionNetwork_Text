# =================================================== #
import torch
from torch import nn 
from torch.nn import functional as F
from torch import FloatTensor as FT
from torch import LongTensor as LT
import numpy as np
# =================================================== #

LOW_SENTINEL = -100000000

class attention_layer(nn.Module):
    def __init__(self, feature_dim):
        
        super(attention_layer, self).__init__()
        self.feature_dim = feature_dim 
        self.W = torch.nn.Parameter(torch.zeros(feature_dim,feature_dim))
        self.b = torch.nn.Parameter(torch.zeros(feature_dim))
        torch.nn.init.normal_(self.W)
        torch.nn.init.normal_(self.b)
        self.context_vector = torch.nn.Parameter(
            data = torch.zeros([feature_dim])
        )
        torch.nn.init.normal_(self.context_vector, mean=0.0, std=1.0)
        return
    
    # ------------------------------------------------
    # Input has shape [batch_size, seq_len, features]
    # ------------------------------------------------
    def forward(self, x, mask):
        global LOW_SENTINEL
        
        x1 = torch.tanh(torch.matmul(x, self.W) + self.b)
        x2 = torch.matmul( x1, self.context_vector)
        mask = LT(mask)
        # ------------------------------------------------
        # Before doing softmax
        # mask out the locations where there is no word
        # ------------------------------------------------
        x2[mask==0] = LOW_SENTINEL
        x2 = x2.reshape([x2.shape[0],x2.shape[1],1])
        x3 = F.softmax(x2,dim=1)
        x4 = x3 * x
        x5 = torch.sum(x4,dim=1,keepdims=False)
        return x5


# =================================================== #
# First layer of HAN
# =================================================== #
class word_encoder(nn.Module):
    def __init__(self, inp_emb_dim, hidden_dim, num_layers= 2):
        super(word_encoder,self).__init__()
        self.bi_gru = torch.nn.GRU(
            input_size = inp_emb_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.att = attention_layer(2*hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        return
    
    
    def get_init_state(self, batch_size):
        num_directions = 2
        # h_0 should be of shape (num_directions * num_layers, batch_size, hidden_dim)
        h_0 = FT(np.zeros ([num_directions * self.num_layers, batch_size, self.hidden_dim]))
        return h_0
    
    # Input is a minibatch of sentences
    # X has shape [ batch_size, seq_len, features]
    def forward(self,x, mask, h_0):
        gru_op = self.bi_gru(x, h_0)[0]
        att_op = self.att(gru_op, mask)
        return att_op
    
        

# =================================================== #
# Second layer of HAN
# =================================================== #
class sentence_encoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers= 2):
        super(sentence_encoder,self).__init__()
        self.bi_gru = torch.nn.GRU(
            input_size = inp_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        self.att = attention_layer(2*hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        return
    
    
    def get_init_state(self, batch_size):
        num_directions = 2
        # h_0 should be of shape (num_directions * num_layers, batch_size, hidden_dim)
        h_0 = FT(np.zeros ([num_directions * self.num_layers, batch_size, self.hidden_dim]))
        return h_0
    
    # Input is a minibatch of sentences
    # X has shape [ batch_size, seq_len, features]
    def forward(self, x, mask, h_0):
        gru_op = self.bi_gru(x, h_0)[0]
        att_op = self.att(gru_op, mask)
        return att_op
    

class HAN_op_layer(nn.Module):
    def __init__(self, inp_dimension, num_classes):
        super(HAN_op_layer,self).__init__()
        if num_classes == 2:
            num_classes = 1
        self.FC = nn.Linear(inp_dimension, num_classes)
        return 
    
    def forward(self,x):
        return self.FC(x)