# General flow: tx_train.yaml -> tx_ft -> tx_val -> tx_test

# tx_train: trains the model.
# tx_ft: uses data-replay to address forgetting. (refer Sec 4.4 in paper)
# tx_val: learns the weibull distribution parameters from a kept aside validation set.
# tx_test: evaluate the final model
# x above can be {1, 2, 3, 4}

# NB: Please edit the paths accordingly.
# NB: Please change the batch-size and learning rate if you are not running on 8 GPUs.
# (if you find something wrong in this, please raise an issue on GitHub)

# Task 1
# Train UDA
rm -r output
python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/t1/t1_uda.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
MODEL.ROI_HEADS.UDA_AUTO_LABELLING False MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 5 OWOD.CUR_INTRODUCED_CLS 20 \
OUTPUT_DIR "./output/t1_uda"

#Train whole net
# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52125' --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 5000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 OWOD.ENABLE_CLUSTERING True OWOD.CLUSTERING.START_ITER 1000 \
# OUTPUT_DIR "./output/t1" MODEL.WEIGHTS "./output/t1_uda/model_final.pth"

#No need to finetune in Task 1, as there is no incremental component.

# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 OWOD.ENABLE_CLUSTERING True \
# OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1/model_final.pth"

# python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 \
# OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "./output/t1/model_final.pth"


# Task 2
# cp -r ./output/t1 ./output/t2
# # Train UDA
# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t2/t2_uda.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING False MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 5 \
# OUTPUT_DIR "./output/t2_uda" MODEL.WEIGHTS "./output/t2/model_final.pth"

# # Train whole net
# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 5000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 \
# OUTPUT_DIR "./output/t2" MODEL.WEIGHTS "./output/t2_uda/model_final.pth"

# cp -r ./output/t2 ./output/t2_ft

# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 \
# OUTPUT_DIR "./output/t2_ft" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"

#python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"

# python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 \
# OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "./output/t2_ft/model_final.pth"


# # Task 3
# cp -r ./output/t2_ft ./output/t3
# # Train UDA
# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t3/t3_uda.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING False MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 5 \
# OUTPUT_DIR "./output/t3_uda" MODEL.WEIGHTS "./output/t3/model_final.pth"

# # Train whole net
# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t3/t3_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 3000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 \
# OUTPUT_DIR "./output/t3" MODEL.WEIGHTS "./output/t3_uda/model_final.pth"

# cp -r ./output/t3 ./output/t3_ft

# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --config-file ./configs/OWOD/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 \
# OUTPUT_DIR "./output/t3_ft" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"

# #python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "./output/t3_ft/model_final.pth"

# python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 \
# OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "./t3_ft/model_final.pth"


# # Task 4
# cp -r ./output/t3_ft ./output/t4

# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t4/t4_uda.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING False MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 5 \
# OUTPUT_DIR "./output/t4_uda" MODEL.WEIGHTS "./output/t4/model_final.pth"

# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52127' --config-file ./configs/OWOD/t4/t4_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 3000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 \
# OUTPUT_DIR "./output/t4" MODEL.WEIGHTS "./output/t4_uda/model_final.pth"

# cp -r ./output/t4 ./output/t4_ft

# python tools/train_net.py --num-gpus 4 --dist-url='tcp://127.0.0.1:52126' --config-file ./configs/OWOD/t4/t4_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 SOLVER.CHECKPOINT_PERIOD 1000 \
# MODEL.ROI_HEADS.UDA_AUTO_LABELLING True MODEL.ROI_HEADS.UDA_POS_FRACTION 1 MODEL.BACKBONE.FREEZE_AT 2 \
# OUTPUT_DIR "./output/t4_ft" MODEL.WEIGHTS "./output/t4_ft/model_final.pth"

# python tools/train_net.py --num-gpus 4 --eval-only --config-file ./configs/OWOD/t4/t4_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 \
# OUTPUT_DIR "./output/t4_final" MODEL.WEIGHTS "./output/t4_ft/model_final.pth"