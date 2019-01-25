# train model
python train.py  configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py --gpus 1

# test model with ground truth
python test.py ./configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py ./work_dirs/faster_rcnn_r50_fpn_1x_voc0712/latest.pth --out work_dirs/result.pkl

# eval model use voc
 python voc_eval.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712/result.pkl ./configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py
