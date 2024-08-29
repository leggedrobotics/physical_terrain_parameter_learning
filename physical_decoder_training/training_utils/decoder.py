import torch
# import rslgym.algorithm.modules as rslgym_module
import pytorch_lightning as pl
from .paradigms import ParadigmFactory
""" Add an option to use only the output from the middle position of the sequence for prediction and loss calculation. """
 
class LightningWrapper:
    @staticmethod
    def create_lightning(args,model_name):
        return DefaultLightning(args,model_name)

class DefaultLightning(pl.LightningModule):
    def __init__(self, args, model_name="model"):
        super().__init__()
        args["model_type"] = model_name
        
        self.decoder=ParadigmFactory.create_para(args)
      
        self.model_name = model_name
        
    def forward(self, exte, latent):  
        return self.decoder(exte, latent)
    
    
    
    def training_step(self, batch, batch_idx):
        latent, priv = batch
                
        batch_size = latent.shape[0]
        if self.decoder.norm:
            self.decoder.priv_normalizer.set_training_mode()
        loss = self.decoder.calc_prediction_loss(latent, priv)

        if self.decoder.paradigm=='rnc':
            self.log(f'{self.model_name}_RNC_backone_train_loss', loss)
        else:
            self.log(f'{self.model_name}_train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        latent, priv = batch
        batch_size = latent.shape[0]
        if self.decoder.norm:
            self.decoder.priv_normalizer.set_validation_mode()
        loss = self.decoder.calc_prediction_loss(latent, priv)

        if self.decoder.paradigm=='rnc':
            self.log(f'{self.model_name}_RNC_backone_val_loss', loss)
        else:
            self.log(f'{self.model_name}_val_loss', loss)
        return loss
    def configure_optimizers(self):
        lr = 0.001
        weight_decay=0.00001
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=weight_decay)
        return optimizer
