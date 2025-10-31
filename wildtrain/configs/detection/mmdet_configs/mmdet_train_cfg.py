_base_ = [
    "./mmdet_configs/two_stage/faster-rcnn_r50_fpn_fcos-rpn_1x.py",
    "./mmdet_configs/dataset.py",
    "./mmdet_configs/scheduler-runtime.py",
]


load_from = None
