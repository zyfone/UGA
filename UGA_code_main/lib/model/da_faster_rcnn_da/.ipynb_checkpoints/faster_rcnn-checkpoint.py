import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn_da.DA import _InstanceDA,_InstanceDA_channel
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
    grad_reverse,
)
from torch.autograd import Variable


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc, da_use_contex, in_channel=4096):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )


        self.da_use_contex = da_use_contex
        if self.da_use_contex:
            if self.lc:
                in_channel += 128
            if self.gc:
                in_channel += 128
        self.RCNN_instanceDA = _InstanceDA(in_channel)

    
        self.RCNN_instanceDA_neg = _InstanceDA(in_channel)
        self.RCNN_instanceDA_pos = _InstanceDA(in_channel)
        self.channel_DA = _InstanceDA_channel(512)

    def forward(
        self,
        im_data,
        im_info,
        im_cls_lb,
        gt_boxes,
        num_boxes,
        target=False,
        eta=1.0,
        weight_value=1.0,
    ):
        if target:
            need_backprop = torch.Tensor([1]).cuda()
            self.RCNN_rpn.eval()
        else:
            need_backprop = torch.Tensor([0]).cuda()
            self.RCNN_rpn.train()

        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        if self.lc:
            d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel)
            # if not target:
            if True:
                _, feat_pixel = self.netD_pixel(base_feat1.detach())
        else:
            d_pixel = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
        base_feat = self.RCNN_base2(base_feat1)
        if self.gc:
            domain_p, _ = self.netD(grad_reverse(base_feat, lambd=eta))
            # if target:
            #     return d_pixel,domain_p#, diff
            _, feat = self.netD(base_feat.detach())
        else:
            domain_p = self.netD(grad_reverse(base_feat, lambd=eta))
            # if target:
            #     return d_pixel,domain_p#,diff



        base_feat_gcn=F.adaptive_avg_pool2d(base_feat,output_size=(1,1))
        # base_feat_gcn=base_feat
        # b,c,w,h=base_feat_gcn.size()
        base_feat_gcn=base_feat_gcn.view(4,-1)
        time_domain=torch.ones_like(base_feat_gcn).cuda()
        time_domain[0,:]=time_domain[0,:]*0
        time_domain[1,:]=time_domain[1,:]*1
        time_domain[2,:]=time_domain[2,:]*2
        time_domain[3,:]=time_domain[3,:]*3
        base_feat_gcn=torch.cat((base_feat_gcn,time_domain),dim=1)

        channel_feat=self.channel_DA(grad_reverse(base_feat_gcn, lambd=eta))



        
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(
            base_feat, im_info, gt_boxes, num_boxes
        )

        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, im_cls_lb)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws,rois_pos,rois_neg = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2))
            )
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        rois_neg = Variable(rois_neg)

        rois_pos = Variable(rois_pos)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "align":
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))


        if cfg.POOLING_MODE == "align":
            pooled_feat_neg = self.RCNN_roi_align(base_feat, rois_neg.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat_neg = self.RCNN_roi_pool(base_feat, rois_neg.view(-1, 5))

        if cfg.POOLING_MODE == "align":
            pooled_feat_pos = self.RCNN_roi_align(base_feat, rois_pos.view(-1, 5))
        elif cfg.POOLING_MODE == "pool":
            pooled_feat_pos = self.RCNN_roi_pool(base_feat, rois_pos.view(-1, 5))



        pooled_feat_neg = self._head_to_tail(pooled_feat_neg)
        pooled_feat_pos = self._head_to_tail(pooled_feat_pos)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        instance_pooled_feat = pooled_feat
        # feat_pixel = torch.zeros(feat_pixel.size()).cuda()
    

        if self.lc:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)
            if self.da_use_contex:
                instance_pooled_feat = torch.cat(
                    (feat_pixel.detach(), instance_pooled_feat), 1
                )
                pooled_feat_pos = torch.cat(
                    (feat_pixel.detach(), pooled_feat_pos), 1
                )
                pooled_feat_neg = torch.cat(
                    (feat_pixel.detach(), pooled_feat_neg), 1
                )


        if self.gc:
            feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat, pooled_feat), 1)
            if self.da_use_contex:
                instance_pooled_feat = torch.cat(
                    (feat.detach(), instance_pooled_feat), 1
                )

                pooled_feat_pos = torch.cat(
                    (feat.detach(), pooled_feat_pos), 1
                )

                pooled_feat_neg = torch.cat(
                    (feat.detach(), pooled_feat_neg), 1
                )
            # compute bbox offset

        
                

        # add instance da
        instance_sigmoid, same_size_label = self.RCNN_instanceDA(
            instance_pooled_feat, need_backprop
        )
        
        
        instance_sigmoid_pos, same_size_label_pos = self.RCNN_instanceDA_pos(
            pooled_feat_pos, need_backprop
        )


        instance_sigmoid_neg, same_size_label_neg = self.RCNN_instanceDA_neg(
            pooled_feat_neg, need_backprop
        )
        instance_loss = nn.BCELoss()

        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)
        

        DA_ins_loss_cls_pos = instance_loss(instance_sigmoid_pos, same_size_label_pos)
        DA_ins_loss_cls_neg = instance_loss(instance_sigmoid_neg, same_size_label_neg)


        if target:
            return d_pixel, domain_p,DA_ins_loss_cls,(DA_ins_loss_cls_pos+DA_ins_loss_cls_neg),channel_feat

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )
            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )
            bbox_pred = bbox_pred_select.squeeze(1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        return (
            rois,
            cls_prob,
            bbox_pred,
            rpn_loss_cls,
            rpn_loss_bbox,
            RCNN_loss_cls,
            RCNN_loss_bbox,
            rois_label,
            d_pixel,
            domain_p,
            DA_ins_loss_cls,
            (DA_ins_loss_cls_pos+DA_ins_loss_cls_neg),
            channel_feat
        )

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
