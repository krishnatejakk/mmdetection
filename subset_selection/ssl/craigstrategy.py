import apricot
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data.sampler import SubsetRandomSampler
import math


class CRAIGStrategy(DataSelectionStrategy):
    """
    Implementation of CRAIG Strategy from the paper :footcite:`mirzasoleiman2020coresets` for supervised learning frameworks.

    CRAIG strategy tries to solve the optimization problem given below for convex loss functions:

    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| x^i - x^j \\|

    In the above equation, :math:`\\mathcal{U}` denotes the training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.

    Since, the above optimization problem is not dependent on model parameters, we run the subset selection only once right before the start of the training.

    CRAIG strategy tries to solve the optimization problem given below for non-convex loss functions:

    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| \\nabla_{\\theta} {L_T}^i(\\theta) - \\nabla_{\\theta} {L_T}^j(\\theta) \\|

    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round,
    and :math:`k` is the budget for the subset. In this case, CRAIG acts an adaptive subset selection strategy that selects a new subset every epoch.

    Both the optimization problems given above are an instance of facility location problems which is a submodular function. Hence, it can be optimally solved using greedy selection methods.
            
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader   
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    selection_type: str
        Type of selection:
         - 'PerClass': PerClass Implementation where the facility location problem is solved for each class seperately for speed ups.
         - 'Supervised':  Supervised Implementation where the facility location problem is solved using a sparse similarity matrix by assigning the similarity of a point with other points of different class to zero.
    """

    def __init__(self, trainloader, valloader, model):
        """
        Constructer method
        """

        super().__init__(trainloader, valloader, model)

        '''self.loss_type = loss_type  # Make sure it has reduction='none' instead of default
        self.device = device
        self.if_convex = if_convex
        self.selection_type = selection_type'''


    def distance(self, x, y, exp=2):
        """
        Compute the distance.
 
        Parameters
        ----------
        x: Tensor
            First input tensor
        y: Tensor
            Second input tensor
        exp: float, optional
            The exponent value (default: 2)
            
        Returns
        ----------
        dist: Tensor
            Output tensor 
        """

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.pow(x - y, exp).sum(2)
        #dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist


    def compute_score(self):
        """
        Compute the score of the indices.

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
        """

        self.compute_gradients()
        
        self.N = self.grads_per_elem.shape[0]

        g_is = self.grads_per_elem
        self.dist_mat = self.distance(g_is, g_is).cpu()

        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()


    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.

        Parameters
        ----------
        idxs: list
            The indices
        
        Returns
        ----------
        gamma: list
            Gradient values of the input indices 
        """

        gamma = [0 for i in range(len(idxs))]
        best = self.dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)
        for i in rep:
            gamma[i] += 1

        return gamma


    def select(self, budget, model_params, optimizer):
        """
        Data selection method using different submodular optimization
        functions.
 
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        optimizer: str
            The optimization approach for data selection. Must be one of
            'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
            'stochastic', 'sample', 'greedi', 'bidirectional'
        
        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints 
        gammas: list
            List containing gradients of datapoints present in greedySet
        """

        gammas = []
        total_greedy_list = []

        idxs = torch.arange(self.N_trn)
        self.compute_score()
        fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                            n_samples=math.ceil(budget/self.trainloader.batch_size), optimizer=optimizer)
        sim_sub = fl.fit_transform(self.dist_mat)
        temp_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
        gammas_temp = self.compute_gamma(temp_list)
        batch_wise_indices = list(self.trainloader.batch_sampler)
        for i in range(len(temp_list)):
            tmp = batch_wise_indices[temp_list[i]]
            total_greedy_list.extend(tmp)
            gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
        return total_greedy_list, gammas
