import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config

#cwh
eps=2e-5
use_global_stats=True
workspace=1024


def residual_unit(data, num_filter, stride, dim_match, name):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
    return sum


def get_resnet_feature(data, units, filter_list):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    #cwh (increase size of feature map)
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    return unit




def _get_rpn(is_train, ft_map, im_info, num_anchors, rpn_label=None, rpn_bbox_target=None, rpn_bbox_weight=None):
    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=ft_map, kernel=(3, 3), pad=(1, 1), num_filter=256, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
    

    

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    if is_train:
        # classification
        rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
        # bounding box regression
        rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
        rpn_bbox_loss_norm = rpn_bbox_loss_ / config.TRAIN.RPN_BATCH_SIZE / config.TRAIN.SAMPLES_PER_BATCH
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_norm, grad_scale=config.TRAIN.RPN_REG_LOSS_WEIGHT)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    cfg1 = config.TRAIN if is_train else config.TEST
    if cfg1.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=cfg1.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg1.RPN_POST_NMS_TOP_N,
            threshold=cfg1.RPN_NMS_THRESH, rpn_min_size=cfg1.RPN_MIN_SIZE)
    else:
        rois = mx.symbol.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=cfg1.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg1.RPN_POST_NMS_TOP_N,
            threshold=cfg1.RPN_NMS_THRESH, rpn_min_size=cfg1.RPN_MIN_SIZE)


    if is_train:
        return rois, rpn_cls_prob, rpn_bbox_loss
    else:
        return rois


def _get_3DCE_head(is_train, ft_map, rois, num_classes):
    num_rfcn_chn = 10
    S = 7
    num_hidden = 2048
    cfg1 = config.TRAIN if is_train else config.TEST
    conv_new_1 = mx.sym.Convolution(data=ft_map, kernel=(1, 1), num_filter=S * S * num_rfcn_chn, name="conv_new_1", lr_mult=3.0)
    conv_new_cat = mx.sym.reshape(conv_new_1, shape=(cfg1.SAMPLES_PER_BATCH, -1, 0,0), name='conv_new_cat')

    # rfcn_cls/rfcn_bbox

    psroipool5 = mx.contrib.sym.PSROIPooling(name='psroipool5', data=conv_new_cat, rois=rois,
                                            group_size=S, pooled_size=S,
                                            output_dim=num_rfcn_chn*config.NUM_IMAGES_3DCE, spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    fc6 = mx.symbol.FullyConnected(name='fc6', data=psroipool5, num_hidden=2048, lr_mult=2.0)
    relu6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu6')

    cls_score = mx.symbol.FullyConnected(name='cls_score', data=relu6, num_hidden=num_classes)
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=relu6, num_hidden=num_classes * 4)
    cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
    bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_classes))

    return cls_score, bbox_pred


def _get_RFCN_head(is_train, ft_map, rois, num_classes):
    num_rfcn_chn = 512
    S = 7
    conv_new_1 = mx.sym.Convolution(data=ft_map, kernel=(1, 1), num_filter=num_rfcn_chn, name="conv_new_1", lr_mult=3.0)
    relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

    rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=S * S * num_classes, name="rfcn_cls", lr_mult=3.0)
    rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=S * S * 4 * num_classes,
                                   name="rfcn_bbox", lr_mult=3.0)

    # rfcn_cls/rfcn_bbox
    psroipool5_cls = mx.contrib.sym.PSROIPooling(name='psroipool5_cls', data=rfcn_cls, rois=rois,
                                                 group_size=S, pooled_size=S,
                                                 output_dim=num_classes, spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    psroipool5_reg = mx.contrib.sym.PSROIPooling(name='psroipool5_reg', data=rfcn_bbox, rois=rois,
                                                 group_size=S, pooled_size=S,
                                                 output_dim=num_classes * 4,
                                                 spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    cls_score = mx.symbol.Pooling(data=psroipool5_cls, global_pool=True, kernel=(S, S), pool_type="avg",
                                  name="cls_score")
    bbox_pred = mx.symbol.Pooling(data=psroipool5_reg, global_pool=True, kernel=(S, S), pool_type="avg",
                                  name="bbox_pred")
    cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
    bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_classes))

    return cls_score, bbox_pred


def _get_Faster_head(is_train, ft_map, rois, num_classes):
    # Fast R-CNN
    S = 7
    num_faster_fc = 2048
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=ft_map, rois=rois, pooled_size=(S, S), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)

    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_faster_fc, name="fc6_small")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=num_faster_fc, name="fc7_small")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    return cls_score, bbox_pred


def get_resnet(is_train, num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    """
    end-to-end train with VGG 16 conv layers with RPN
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    # data
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    if is_train:
        gt_boxes = mx.symbol.Variable(name="gt_boxes")
        rpn_label = mx.symbol.Variable(name='label')
        rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
        rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    
    #cwh
    units=(2, 2, 2, 2)
    filter_list=(64,128,256,512)
    relu5_3 = get_resnet_feature(data, units=units, filter_list=filter_list)

    # RPN
    if is_train:
        rois, rpn_cls_prob, rpn_bbox_loss = _get_rpn(
            is_train, relu5_3, im_info, num_anchors, rpn_label, rpn_bbox_target, rpn_bbox_weight)
        # ROI proposal target
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes, op_type='proposal_target',
                                 num_classes=num_classes, batch_images=config.TRAIN.SAMPLES_PER_BATCH,
                                 batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
        rois, label, bbox_target, bbox_weight = group

    else:
        rois = _get_rpn(is_train, relu5_3, im_info, num_anchors)

    # RCNN head
    cls_score, bbox_pred = eval('_get_'+config.FRAMEWORK+'_head')(is_train, relu5_3, rois, num_classes)

    # loss and output
    if is_train:
        cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
        bbox_loss_norm = bbox_loss_ / config.TRAIN.BATCH_ROIS / config.TRAIN.SAMPLES_PER_BATCH
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_norm, grad_scale=config.TRAIN.RCNN_REG_LOSS_WEIGHT)

        # reshape output
        label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.SAMPLES_PER_BATCH, -1), name='label_reshape')
        cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.SAMPLES_PER_BATCH, -1, num_classes), name='cls_prob_reshape')
        bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.SAMPLES_PER_BATCH, -1, 4 * num_classes), name='bbox_loss_reshape')
        group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    else:
        cls_prob = mx.symbol.softmax(name='cls_prob', data=cls_score)

        # reshape output
        batchsize = config.TEST.SAMPLES_PER_BATCH
        cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(batchsize, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(batchsize, -1, 4 * num_classes), name='bbox_pred_reshape')
        group = mx.symbol.Group([rois, cls_prob, bbox_pred])

    return group

