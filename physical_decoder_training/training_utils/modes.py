import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import legged_gym.modules as rslgym_module

class ModeFactory:
    @staticmethod
    def create_mode(config):
        mode_type = config['mode']
        if mode_type == "base":
            return BaseMode(config)
        elif mode_type == "parallel":
            return ParallelMode(config)
        elif mode_type == "mlp":
            return MlpMode(config)
        elif mode_type == "parallelnoattention":
            return ParallelnoattMode(config)
        elif mode_type == "parallelnogru":
            return ParallelnogruMode(config)
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")



# temporal-wise attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x,mask=None):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        if mask is not None:
            scores += mask 
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class Mode(nn.Module):
    def __init__(self, config):
        super(Mode, self).__init__()
        if not isinstance(config, dict):
            raise ValueError("Config must be a dict.")
        self.config = config
        # Set default values and override with any values provided in config
        self.sens_size = config['sens_size']
        self.priv_size = config['priv_size']
        self.hidden_to_pred_shape = config['hidden_to_pred_shape']
        self.hidden_to_gate_shape = config['hidden_to_gate_shape']
        self.gate_to_pred_shape = config['gate_to_pred_shape']
        self.hidden_size = config['hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.seq_length = config['seq_length']
        self.mode = config['mode']
        self.is_training = config['is_training']
        self.model_type = config['model_type']
        
        self.reset_hidden=config.get('reset_hidden_each_epoch', True)
        self.l1_alpha = config.get('l1_alpha', 0.2)
        self.activation_fn = config.get('activation_fn', "LeakyReLU")
        if self.activation_fn == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU
        self.device = config.get('device', 'cpu')
        self.paradigm = config.get('paradigm', None)
        self.backbone = config.get('backbone', None)
        self.transformer_config = config.get('transformer_config', None)
        # self.rnc = config.get('rnc', False)
        self.norm = config.get('norm', False)
        self.norm_in = config.get('norm_in', False)
        self.middle_only = config.get('middle_only', False)
        self.dropout_rate=config.get('dropout_rate',0.0)

        # Calculate derived properties
        self.encoder_input_dim = self.sens_size 

    
    def forward(self, input):
        raise NotImplementedError

    def forward_rnn(self, x, hidden):
        output, next_hidden = self.rnn(x, hidden)
        output = output.permute(1, 0, 2)
        return output, next_hidden
    
    def init_hidden(self, batch_size):
        """
        Return new latent state.
        batch_size is the number of parallel environments being used.
        """
        h0 = torch.zeros(self.num_rnn_layers , batch_size, self.hidden_size).to(self.device)
        return h0
    
        

class BaseMode(Mode):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_to_pred = rslgym_module.MLP(
                self.hidden_to_pred_shape, self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
            )
        self.rnn = nn.GRU(self.encoder_input_dim , self.hidden_size, self.num_rnn_layers,bidirectional=False)
        self.attention=SelfAttention(self.hidden_size)
        self.mask=None
    
    def forward(self, x,hidden):
        x = x.permute(1, 0, 2)
        if self.reset_hidden or hidden is None:
            hidden=self.init_hidden(x.shape[1])
        hidden=hidden.detach()
        self.output, next_hidden = self.forward_rnn(x, hidden)
        self.output_weighted=self.attention(self.output)
        pred = self.hidden_to_pred(self.output_weighted)
        
        return pred, next_hidden
    
class ParallelnoattMode(Mode):
    def __init__(self, config):
        super().__init__(config)
        self.alpha=None
        self.pro_size=self.sens_size-208
        self.hidden_to_pred = rslgym_module.MLP(
                self.hidden_to_pred_shape, self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
            )
        self.rnn = nn.GRU(self.encoder_input_dim , self.hidden_size, self.num_rnn_layers,bidirectional=False)
        self.rnn_pro = nn.GRU(self.pro_size, self.hidden_size, self.num_rnn_layers,bidirectional= False)
        self.hidden_to_pred = rslgym_module.MLP(
            self.hidden_to_pred_shape, self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.hidden_to_gate = rslgym_module.MLP(
            self.hidden_to_gate_shape,self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.hidden_to_pred_exte = rslgym_module.MLP(
            self.hidden_to_pred_shape,self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )        
    
    def forward(self, x,hidden):
        x = x.permute(1, 0, 2) # convert for rnn input
        if self.reset_hidden or hidden is None:
            hidden=self.init_hidden(x.shape[1])
        hidden=hidden.detach()
        # extract sens [:,:,-208:] as exteroceptive input and pass through the encoder. THen concantenate the embedding with sens[:,:,:133]
        exte_input=x[:,:,-208:]
        pro_input=x[:,:,:self.pro_size]
        self.output, next_hidden = self.forward_rnn(x, hidden)
        self.output_weighted=self.output
        pro_output,_=self.rnn_pro(pro_input,hidden)
        pro_output=pro_output.permute(1, 0, 2)
        pro_output_weighted=pro_output
        pro_pred=self.hidden_to_pred_exte(pro_output_weighted)
        self.alpha=F.sigmoid(self.hidden_to_gate(self.output_weighted))
        filtered_pro_output=self.alpha*pro_pred
        pred=filtered_pro_output+self.hidden_to_pred(self.output_weighted)
        
        return pred, next_hidden    
    
class ParallelnogruMode(Mode):
    def __init__(self, config):
        super().__init__(config)
        self.alpha=None
        self.pro_size=self.sens_size-208
        self.hidden_to_pred = rslgym_module.MLP(
                self.hidden_to_pred_shape, self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
            )
        self.attention=SelfAttention(self.hidden_size)
        self.hidden_to_pred = rslgym_module.MLP(
            self.hidden_to_pred_shape, self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.hidden_to_gate = rslgym_module.MLP(
            self.hidden_to_gate_shape,self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.hidden_to_pred_exte = rslgym_module.MLP(
            self.hidden_to_pred_shape,self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.in_to_hidden = rslgym_module.MLP(
            [256], self.activation_fn, self.sens_size, self.hidden_size, init_scale=1.0 / np.sqrt(2)
        )
        self.in_to_hidden_exte = rslgym_module.MLP(
            [128], self.activation_fn, self.pro_size, self.hidden_size, init_scale=1.0 / np.sqrt(2)
        )
        self.attention_exte=SelfAttention(self.hidden_size)
        self.mask=None
    
    def forward(self, x,hidden):
        if self.reset_hidden or hidden is None:
            hidden=self.init_hidden(x.shape[1])
        hidden=hidden.detach()
        # extract sens [:,:,-208:] as exteroceptive input and pass through the encoder. THen concantenate the embedding with sens[:,:,:133]
        exte_input=x[:,:,-208:]
        pro_input=x[:,:,:self.pro_size]
        self.output, next_hidden = self.in_to_hidden(x), hidden
        self.output_weighted=self.attention(self.output,self.mask)
        pro_output=self.in_to_hidden_exte(pro_input)
        pro_output_weighted=self.attention_exte(pro_output,self.mask)
        pro_pred=self.hidden_to_pred_exte(pro_output_weighted)
        self.alpha=F.sigmoid(self.hidden_to_gate(self.output_weighted))
        filtered_pro_output=self.alpha*pro_pred
        pred=filtered_pro_output+self.hidden_to_pred(self.output_weighted)
        
        return pred, next_hidden

class ParallelMode(BaseMode):
    def __init__(self, config):
        super().__init__(config)
        self.alpha=None
        self.pro_size=self.sens_size-208
        self.rnn_pro = nn.GRU(self.pro_size, self.hidden_size, self.num_rnn_layers,bidirectional= False)
        self.hidden_to_pred = rslgym_module.MLP(
            self.hidden_to_pred_shape, self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.hidden_to_gate = rslgym_module.MLP(
            self.hidden_to_gate_shape,self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.hidden_to_pred_exte = rslgym_module.MLP(
            self.hidden_to_pred_shape,self.activation_fn, self.hidden_size, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
        self.attention_exte=SelfAttention(self.hidden_size)
        self.mask=None
        
    def generate_mask(self,seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
    
    def forward(self, x,hidden):
        x = x.permute(1, 0, 2) # convert for rnn input
        if self.reset_hidden or hidden is None:
            hidden=self.init_hidden(x.shape[1])
        hidden=hidden.detach()
        # extract sens [:,:,-208:] as exteroceptive input and pass through the encoder. THen concantenate the embedding with sens[:,:,:133]
        exte_input=x[:,:,-208:]
        pro_input=x[:,:,:self.pro_size]
        self.output, next_hidden = self.forward_rnn(x, hidden)
        self.output_weighted=self.attention(self.output,self.mask)
        pro_output,_=self.rnn_pro(pro_input,hidden)
        pro_output=pro_output.permute(1, 0, 2)
        pro_output_weighted=self.attention_exte(pro_output,self.mask)
        pro_pred=self.hidden_to_pred_exte(pro_output_weighted)
        self.alpha=F.sigmoid(self.hidden_to_gate(self.output_weighted))
        filtered_pro_output=self.alpha*pro_pred
        pred=filtered_pro_output+self.hidden_to_pred(self.output_weighted)
        
        return pred, next_hidden

class MlpMode(Mode):
    def __init__(self, config):
        super().__init__(config)
        self.in_to_pred = rslgym_module.MLP(
            self.hidden_to_pred_shape, self.activation_fn, self.encoder_input_dim, self.priv_size, init_scale=1.0 / np.sqrt(2)
        )
    
    def forward(self, x,hidden):
        pred=self.in_to_pred(x)
        return pred, hidden

        