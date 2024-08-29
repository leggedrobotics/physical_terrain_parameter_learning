import torch.nn as nn
import numpy as np
import torch
import legged_gym.modules as rslgym_module
from .modes import ModeFactory

class BinAccuracy:
    def __init__(self, ord_num, model_type, discretization="SID",shift_SID=0.01):
        self.ord_num = ord_num
        self.model_type = model_type
        self.discretization = discretization
        self.shift=shift_SID
        if self.model_type == "fric":
            self.alpha, self.beta = 0, 1
        elif self.model_type == "stiff":
            self.alpha, self.beta = 1, 10
        else:
            raise ValueError("Unknown model type")

    def _get_label(self, gt):
        if self.discretization == "SID":
            shift=torch.tensor(self.shift).to(gt.device)
            alpha = torch.tensor(self.alpha +self.shift).to(gt.device)
            beta=torch.tensor(self.beta+self.shift).to(gt.device)
            gt=gt+shift
            label =  torch.log(gt / alpha) / torch.log(beta / alpha)*self.ord_num 
        elif self.discretization == "UD":
            label =  (gt - self.alpha) / (self.beta - self.alpha)*self.ord_num
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization}")
        label = label.clamp(max=self.ord_num-1,min=0)
        label = label.long()
        return label

    def calculate_accuracy(self, pred_or_prob, gt, greater_eq_probs=None):
      
        pred_label = self._get_label(pred_or_prob)
        gt_label = self._get_label(gt)
        pred_label=pred_label.to(gt.device)
        # Initialize a dictionary to store probability vectors of false assignments
        assignments_probs = {i: {j: [] for j in range(self.ord_num) } for i in range(self.ord_num)}

        accuracies = []
        # Initialize confusion matrix
        confusion_matrix = torch.zeros(self.ord_num, self.ord_num, dtype=torch.int)
        for i in range(self.ord_num):
            if self.discretization == "UD":
                bin_start = self.alpha + i * (self.beta - self.alpha) / self.ord_num
                bin_end = self.alpha + (i + 1) * (self.beta - self.alpha) / self.ord_num
            else:  # SID
                shift=torch.tensor(self.shift).to(gt.device)
                shifted_alpha = torch.tensor(self.alpha + self.shift).to(gt.device)
                shifted_beta = torch.tensor(self.beta + self.shift).to(gt.device)
                bin_start = torch.exp(torch.log(shifted_alpha) + torch.log(shifted_beta / shifted_alpha) * i / self.ord_num) - shift
                bin_end = torch.exp(torch.log(shifted_alpha) + torch.log(shifted_beta / shifted_alpha) * (i + 1) / self.ord_num) - shift

            correct = (pred_label == i) & (gt_label == i)
            total = (gt_label == i)

            # Handle empty bins
            if total.float().sum() == 0:
                accuracies.append(float('nan'))
            else:
                accuracy = correct.float().sum() / total.float().sum()
                accuracies.append(accuracy.item())
            # Extract the predictions corresponding to the current gt bin
            mask = (gt_label == i)
            binned_preds_for_current_gt_bin = pred_label[mask]

            # Update the confusion matrix
            for j in range(self.ord_num):
                confusion_matrix[i, j] = (binned_preds_for_current_gt_bin == j).sum()
            # Check for false assignments and extract their probability vectors
            if greater_eq_probs is not None:
                for j in range(self.ord_num):              
                    assignments_mask = (gt_label == i) & (pred_label == j)
                    assignments_indices = torch.nonzero(assignments_mask).squeeze()
                    AA=greater_eq_probs[assignments_indices[:, 0], assignments_indices[:, 1], assignments_indices[:, 2]]
                    assignments_probs[i][j].extend(AA.tolist())
            print(f"Bin {i}: Range [{bin_start:.2f}, {bin_end:.2f}] - Accuracy: {accuracies[-1]*100:.2f}%")
        # Display the confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix)
        if greater_eq_probs is not None:
            # Calculate and display the average probability vectors for false assignments
            print("\nAverage Probability vectors for assignments:")
            for i in range(self.ord_num):
                for j in range(self.ord_num):
                    # Before the problematic line
                    SS= assignments_probs[i][j]
                    avg_probs = [sum(ch)/len(ch) for ch in zip(*assignments_probs[i][j])]
                    print(f"From bin {i} to bin {j}: {avg_probs}")
        return accuracies,confusion_matrix
    
class ParadigmFactory:
    @staticmethod
    def create_para(config):
        paradigm=config['paradigm']
        if paradigm is None:
            return Default(config)
        else:
            raise ValueError("Unknown paradigm type")

class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.input_type = config['input_type']
        
        
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
        # Calculate derived properties
        self.encoder_input_dim = self.sens_size 
        if self.model_type=="fric":
            self.discretization = "SID"
            self.ord_num=5
            self.shift=0.2
        else:
            self.discretization = "SID"
            self.ord_num=5
            self.shift=1.0
        self.bin_accuracy_cal=BinAccuracy(self.ord_num,model_type=self.model_type,discretization=self.discretization,shift_SID=self.shift)
        self.ranges_weight = None
        self.interval=None
        self.model=None
        self.priv_normalizer=None
        self.predhead=None
        self.hidden=None
        self.prob=False
        self.exact_middle_step=None
        
    def forward(self, latent, hidden=None):
        # TODO: implement this
        if self.norm_in:
            pass
            raise NotImplementedError
        pred, next_hidden = self.model(latent,hidden)
        if self.backbone:
            pred=self.predhead(pred)
        """ if self.middle_only: extract the middle position pred into a single dimension (keep sequence formatt but=1) """
        if self.middle_only:
            if self.exact_middle_step is None:
                # self.exact_middle_step=self.seq_length//2
                self.exact_middle_step=-1
            pred=pred[:,self.exact_middle_step,:].unsqueeze(1)
        return pred, next_hidden
    
    def get_rnn_output(self):
        return self.output

    def get_latent(self):
        return self.latent

    def reset_parameters(self):
        kaimin_normal = torch.nn.init.kaiming_normal_
        xavier_normal = torch.nn.init.xavier_normal_
        gain = nn.init.calculate_gain("relu")
        orthogonal = torch.nn.init.orthogonal_

        def weights_init():
            def fn(m):
                classname = m.__class__.__name__
                if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                    kaimin_normal(m.weight.data)
                    try:
                        m.bias.data.fill_(0)
                    except:
                        pass

                if classname.find("PFGRU") != -1:
                    xavier_normal(m.fc_z.weight.data)
                    xavier_normal(m.fc_r.weight.data)
                    kaimin_normal(m.fc_n.weight.data, nonlinearity="relu")
                    orthogonal(m.fc_obs.weight.data, gain=gain)
                    m.fc_z.bias.data.fill_(0)
                    m.fc_r.bias.data.fill_(0)
                    m.fc_n.bias.data.fill_(0)
                    m.fc_obs.bias.data.fill_(0)

            return fn

        self.apply(weights_init())
        
    def calc_prediction_loss(self, output,  priv):
        raise NotImplementedError
    
    def recon_loss(self, output, true_state,loss_w_in=None):
        batch_size = true_state.shape[0]

        
        if loss_w_in is not None:
            # weights based on data balance
            loss_w = loss_w_in
        else:
            loss_w = torch.ones(batch_size, true_state.shape[1], true_state.shape[2]).to(self.device)
        loss_l2 = self.weighted_mse(output, true_state, loss_w)
        loss_l1 = self.weighted_l1(output, true_state, loss_w)
        loss = loss_l2 + self.l1_alpha * loss_l1
        return loss
    
    def inout_selection(self, input_data, output_data=None):
        """ 
        Here we select the input and output data based on the input type and output type of the decoder.
        """
        input_type = self.input_type
        if input_type == "hidden":
            input_data = input_data[..., -100:]  # Extract the last 100 columns
        elif input_type == "pro+exte":
            input_data = input_data[..., :341]  # Extract the first 341 columns
        elif input_type == "pro":
            input_data = input_data[..., :133]  # Extract the first 133 columns
            
        if output_data is not None:
            output_type=self.model_type
            if output_type == "fric":
                output_data = output_data[..., :4]
            elif output_type == "stiff":
                output_data = output_data[..., 4:]
            else:
                raise ValueError(f"Unknown output type: {output_type}")
            return input_data, output_data
        else:
            return input_data
    
    def get_unnormalized_recon(self, latent,hidden):
        latent = self.inout_selection(latent)
        priv_recon,next_hidden= self.forward( latent,hidden)
        # print(self.priv_normalizer._mean)
        # return self.priv_normalizer.inverse(priv_recon), self.exte_normalizer.inverse(exte_recon)
        if self.norm:
            return self.priv_normalizer.inverse(priv_recon),next_hidden
        else:
            return priv_recon,next_hidden
    
    def weighted_mse(self, a, b, w):
        # w = w.reshape(a.shape[0], 1)
        return (w * (a - b) ** 2).mean()

    def weighted_l1(self, a, b, w):
        # w = w.reshape(a.shape[0], 1)
        return (w * torch.abs(a - b)).mean()

    def get_loss(self):
        return self.loss
    
    def set_range_weight(self, data,use_weight):

        if self.model_type == "fric":
            interval = 0.1
            max_range=1
            min_range=0
        elif self.model_type == "stiff":
            interval = 1
            max_range=10
            min_range=1
        else:
            raise ValueError("Unknown model type")
        # Convert tuple of tensors to a single numpy array
        data = torch.cat(data, dim=0).cpu().numpy()
        # Adjust range if needed
        bins = np.arange(min_range, max_range+ interval, interval)
        hist, _ = np.histogram(data, bins=bins)
        self.interval=interval
        # Compute weights based on the inverse of frequency
        if use_weight:
            weights = 1.0 / (hist + 1e-8)  # Adding epsilon to avoid division by zero
        else:
            weights=0.0*hist+1.0
        weights /= weights.sum()
        self.ranges_weight = torch.tensor(weights).to(self.device)
        
    def generate_mask(self,seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

class Default(Base):
    def __init__(self, config):
        super().__init__(config)   
        self.model=ModeFactory.create_mode(config)
        if self.norm:
        #     self.priv_normalizer = rslgym_module.EmpiricalNormalization(
        #     [self.priv_size ], until=1e8, clip_threshold=10, is_training=self.is_training,batch_axis=(0,1)
        # )       
            self.priv_normalizer = rslgym_module.EmpiricalNormalization(
            [self.seq_length if not self.middle_only else 1 ,self.priv_size ], until=1e8, clip_threshold=10
        )
        # self.model.mask=self.generate_mask(self.seq_length)
        self.reset_parameters()
    
    def calc_prediction_loss(self, latent, priv):
        # start_time = time.time()
        latent = self.inout_selection(latent)
        output,self.hidden= self.forward( latent,self.hidden)
        # end_time = time.time()  # Capture the end time

        # Calculate and print the duration
        # duration = end_time - start_time
        # print(f"Running time: {duration} seconds")
        if output.shape[1]!=priv.shape[1]:
            if self.middle_only and output.shape[1]==1 :
                priv=priv[:,self.exact_middle_step,:].unsqueeze(1)
            elif self.mode=='transformer' or self.mode=='cnn':
                priv=priv[:,-1,:]
            else:
                raise ValueError("The output and priv should have the same sequence length.")
        priv_recon= output
        if self.norm:
            normalized_priv = self.priv_normalizer(priv)
        else:
            normalized_priv=priv
        loss_w=None
        if self.ranges_weight is not None and self.interval is not None:
            indices = (priv / self.interval).long()
            indices = torch.clamp(indices, 0, len(self.ranges_weight) - 1)  # ensure indices are within range
            loss_w = self.ranges_weight[indices]
        self.priv_loss = self.recon_loss(priv_recon, normalized_priv,loss_w)
        # priv_recon_not_scaled=self.priv_normalizer.inverse(priv_recon)
        # self.priv_loss_not_scaled = self.recon_loss(priv_recon_not_scaled, priv,loss_w)
        self.loss = self.priv_loss    
        return self.loss