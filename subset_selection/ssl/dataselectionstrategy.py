import numpy as np
import torch
import torch.nn.functional as F


class DataSelectionStrategy(object):
    """ 
    Implementation of Data Selection Strategy class which serves as base class for other 
    dataselectionstrategies for supervised learning frameworks.
    """

    def __init__(self, trainloader, valloader,model):
        """
        Constructer method
        """
        
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.dataset)
        self.N_val = len(valloader.dataset)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        

    def select(self, budget, model_params):
        pass

    def compute_gradients(self, valid=False):

        self.grads_per_elem = self.model.module.compute_gradients(self.trainloader,\
             self.model.device_ids)
        self.val_grads_per_elem = self.model.module.compute_gradients(self.valloader,\
             self.model.device_ids)
       
    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """

        self.model.load_state_dict(model_params)
