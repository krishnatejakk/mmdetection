from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.parallel.scatter_gather import scatter_kwargs
import torch 


@DETECTORS.register_module()
class RepPointsDetector(SingleStageDetector):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)

    def compute_gradients (self, data_loader, device_ids,grouping=1):

        for i, data_batch in enumerate(data_loader):
            inputs, kwargs = scatter_kwargs(data_batch, {}, device_ids, dim=0)
            losses = self(**inputs[0], freeze=True)
            loss = losses['loss_cls'] + losses['loss_pts_init']+losses['loss_pts_refine']

            grads = torch.autograd.grad(loss, self.bbox_head.parameters(), allow_unused=True, retain_graph=True)
                        
            if i % grouping == 0:
                tmp_grads = torch.cat((grads[0].view(1, -1),grads[1].view(1, -1),\
                     grads[2].view(1, -1)), dim=1)
            else:
                tmp_grads += torch.cat((grads[0].view(1, -1),grads[1].view(1, -1),\
                     grads[2].view(1, -1)), dim=1)

                if (i+1) % grouping == 0:
                    if (i+1) // grouping == 1:
                        grads_per_elem = tmp_grads/grouping
                    else:
                        grads_per_elem = torch.cat((grads_per_elem, tmp_grads/grouping), dim=0)

        return grads_per_elem