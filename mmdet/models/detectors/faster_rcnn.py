from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmcv.parallel.scatter_gather import scatter_kwargs
import torch

@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def compute_gradients(self, data_loader, device_ids, combine_batches=1):

        for i, data_batch in enumerate(data_loader):
            inputs, kwargs = scatter_kwargs(data_batch, {}, device_ids, dim=0)
            losses = self(**inputs[0], freeze=True)
            loss = losses['loss_cls'] + losses['loss_bbox']
            grads = torch.autograd.grad(loss, self.roi_head.bbox_head.parameters(), allow_unused=True,
                                        retain_graph=True)

            if i % combine_batches == 0:
                if i == combine_batches:
                    grads_per_elem = cmb_grads / combine_batches
                elif i != 0:
                    grads_per_elem = torch.cat((grads_per_elem, (cmb_grads / combine_batches)), dim=0)
                cmb_grads = torch.cat((grads[1].view(1, -1), grads[3].view(1, -1)), dim=1)
            else:
                tmp_grads = torch.cat((grads[1].view(1, -1), grads[3].view(1, -1)), dim=1)
                cmb_grads = cmb_grads + tmp_grads

        grads_per_elem = torch.cat((grads_per_elem, cmb_grads / ((i % combine_batches) + 1)), dim=0)
        return grads_per_elem

