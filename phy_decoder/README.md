# Stand-alone Pre-trained Physical Decoder (Friction+Stiffness)

This two decoders use sequence data as input and output a physical parameters sequence (friction or stiffness). The main architecture is GRU+Self-Attention with a parallel structure.

## Getting Started
### Installing
First, clone this repository to your local machine and install the dependencies.
```shell
cd phy_decoder/

# Install the dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Usage
The model_pth is automatically loaded from the package folder. 

**Attention**: Only env_num=1 is tested now. If you want to use env_num>1, you may need to modify the code a little bit.

```python
from phy_decoder import initialize_models, prepare_padded_input, RNNInputBuffer
# Initializing
step = 0
env_num=1

fric_predictor, stiff_predictor, predictor_cfg = initialize_models()

fric_hidden = fric_predictor.init_hidden(env_num)
stiff_hidden = stiff_predictor.init_hidden(env_num)

input_buffers = {0: RNNInputBuffer()}

# Main loop
while True:
    step += 1 # Pre-increment to 1 for first step not 0
    input_data=obs[:,:341] # Input is batch_size (env_num)x341dim(prop. + ext.)
    padded_inputs = prepare_padded_input(input_data, input_buffers, step, env_num)
    padded_input = torch.stack(padded_inputs, dim=0) # reshape to env_num x num_timesteps x 341
    
    if predictor_cfg['reset_hidden_each_epoch']:
        fric_hidden = fric_predictor.init_hidden(env_num)
        stiff_hidden = stiff_predictor.init_hidden(env_num)
    
    with torch.no_grad():
        # Predict using the friction predictor
        fric_pred, fric_hidden = fric_predictor.get_unnormalized_recon(padded_input, fric_hidden)           
        # Predict using the stiffness predictor
        stiff_pred, stiff_hidden = stiff_predictor.get_unnormalized_recon(padded_input, stiff_hidden)

        # fric_pred, stiff_pred are dimension: env_num x num_timesteps x 4 (number of feet)

    input_buffers[0].add(input_data[0].unsqueeze(0))

# You may use the last step of output sequence as the prediction
fric_pred=fric_pred[:,-1,:]
stiff_pred=stiff_pred[:,-1,:]
```
