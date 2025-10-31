_base_ = [
    "./faster-rcnn_r50_fpn.py",
]

model = dict(
    # copied from configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
    neck=dict(
        start_level=1,
        add_extra_convs="on_output",  # use P5
        relu_before_extra_convs=True,
    ),
    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type="FCOSHead",
        num_classes=1,  # num_classes = 1 for rpn, if num_classes > 1, it will be set to 1 in TwoStageDetector automatically
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="IoULoss", loss_weight=1.0),
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
    ),
    roi_head=dict(  # update featmap_strides due to the strides in neck
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])
    ),
)
