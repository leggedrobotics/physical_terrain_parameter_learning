# Physical param decoder (friction+stiffness).

This two decoders use a sequence data as input and output a sequence of physical parameters (friction or stiffness). The main architecture is GRU+Self-Attention with a parallel structure.

## Getting Started

These structions will help you install this decoder package and use them in your code.

### Installing
First, clone this repository to your local machine.

    pip install -r requirements.txt

Then, install the package.

    pip install -e .

### Usage
This example is implemented in a class. You can also use it without using a class.
The model_pth is automatically loaded from the package folder. 

**Attention**: Only env_num=1 is tested now. If you want to use env_num>1, you may need to modify the code a little bit.

```python
from Phy_Decoder import initialize_models, prepare_padded_input, RNNInputBuffer
# initializing
self.step = 0
self.env_num=1
self.fric_predictor,self.stiff_predictor,self.predictor_cfg=initialize_models()
self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
self.input_buffers = {0: RNNInputBuffer()}
....
# in the main loop
self.step += 1 #definitely use 1 as the first step not 0
input_data=obs[:,:341] #input is batch_size(env_num)x341 dim(pro+exte info)
padded_inputs = prepare_padded_input(input_data, self.input_buffers, self.step, self.env_num)    
padded_input = torch.stack(padded_inputs, dim=0)
if self.predictor_cfg['reset_hidden_each_epoch']:
    self.fric_hidden = self.fric_predictor.init_hidden(self.env_num)
    self.stiff_hidden = self.stiff_predictor.init_hidden(self.env_num)
with torch.no_grad():
    # Predict using the friction predictor
    fric_pred, self.fric_hidden = self.fric_predictor.get_unnormalized_recon(padded_input, self.fric_hidden)           
    # Predict using the stiffness predictor
    stiff_pred, self.stiff_hidden = self.stiff_predictor.get_unnormalized_recon(padded_input, self.stiff_hidden)
self.input_buffers[0].add(input_data[0].unsqueeze(0))

# you may use the last step of output sequence as the prediction
fric_pred=fric_pred[:,-1,:]
stiff_pred=stiff_pred[:,-1,:]
```
