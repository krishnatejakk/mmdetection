# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import torch
import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import numpy as np
from mmdet.datasets import (build_dataloader,
                            replace_ImageToTensor)

from subset_selection.ssl import *
import numpy as np


@RUNNERS.register_module()
class CustomEpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    #def subset(self, data_loader, **kwargs):
    #    grads_per_batch = self.model.module.compute_gradients(data_loader, self.model.device_ids)

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, val_dataloader, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        cfg = kwargs['cfg']

        if cfg.dss_strategy == 'GradMatchPB':
            setf_model = OMPGradMatchStrategy(data_loaders, val_dataloader,self.model,\
                 valid=False, lam=0.25, eps=1e-10)

        elif cfg.dss_strategy == 'CRAIG':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(data_loaders, val_dataloader,self.model)

        elif cfg.dss_strategy == 'CRAIGPB':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(data_loaders, val_dataloader,self.model)

        elif cfg.dss_strategy == 'CRAIG-Warm':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(data_loaders, val_dataloader,self.model)

            kappa_iterations = int(cfg.kappa * max_iteration)
            #full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])

        elif cfg.dss_strategy == 'CRAIGPB-Warm':
            # CRAIG Selection strategy
            setf_model = CRAIGStrategy(data_loaders, val_dataloader,self.model)
            kappa_iterations = int(cfg.kappa * max_iteration)
            #full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])

        elif cfg.dss_strategy == 'Random':
            # Random Selection strategy
            setf_model = RandomStrategy(data_loaders, online=False)

        elif cfg.dss_strategy == 'Random-Online':
            # Random-Online Selection strategy
            setf_model = RandomStrategy(data_loaders, online=True)

        elif cfg.dss_strategy == 'GradMatchPB-Warm':
            # OMPGradMatch Selection strategy
            setf_model = OMPGradMatchStrategy(data_loaders, val_dataloader,self.model,\
                valid=False, lam=0.25, eps=1e-10)
            kappa_iterations = int(cfg.kappa * max_iteration)

        '''elif cfg.dss_strategy == 'Random-Warm':
            kappa_iterations = int(cfg.kappa * max_iteration)

        elif cfg.dss_strategy == 'Full':
            kappa_iterations = int(0.01 * max_iteration)

        elif cfg.dss_strategy == 'GLISTER':
            # GLISTER Selection strategy
            setf_model = GLISTERStrategy(data_loaders, val_dataloader,self.model, teacher_model1, ssl_alg, consistency_nored,
                    cfg.lr, device, num_classes, False, 'Stochastic', r=int(bud))

        elif cfg.dss_strategy == 'GLISTER-Warm':
            # GLISTER Selection strategy
            setf_model = GLISTERStrategy(data_loaders, val_dataloader,self.model, teacher_model1, ssl_alg, consistency_nored,
                    cfg.lr, device, num_classes, False, 'Stochastic', r=int(bud))
            kappa_iterations = int(cfg.kappa * max_iteration)
            #full_epochs = round(kappa_epochs * self.configdata['dss_strategy']['fraction'])'''



        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        N =  len(data_loaders[0].dataset)
        bud = int(cfg.fraction * N)
        #Initial Random Subset of the Dataset
        start_idxs = numpy.random.choice(N, size=bud, replace=False)
        dss_subset = Subset(data_loaders[0].dataset, start_idxs)
        gammas = torch.ones(len(start_idxs), device="cuda")

        data_ss_loaders = build_dataloader(dss_subset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=kwargs['distributed'],
            shuffle=False,
            seed=cfg.seed) 

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'subset':
                epochs = cfg.select_every
        
    
        while self.epoch < self._max_epochs:

            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    if mode == 'train':
                        epoch_runner(data_ss_loaders, **kwargs)
                    elif mode == 'subset':
                        subset_idxs,gammas = setf_model.select(bud)
                        dss_subset = Subset(data_loaders[0].dataset, subset_idxs)

                        data_ss_loaders = build_dataloader(dss_subset,
                            cfg.data.samples_per_gpu,
                            cfg.data.workers_per_gpu,
                            # cfg.gpus will be ignored if distributed
                            len(cfg.gpu_ids),
                            dist=kwargs['distributed'],
                            shuffle=False,
                            seed=cfg.seed) 
                        
                

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class CustomRunner(CustomEpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
