SCRIPT=haftrack
CONFIG=vit_cte_haf_vtuav
nohup python tracking/train.py \
    --script ${SCRIPT} \
    --config ${CONFIG} \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 1 \
    --use_wandb 0 \
    --use_lmdb 0 \
    --vis_gpus 1 \
    >./experiments/${SCRIPT}/train_log/${SCRIPT}-${CONFIG}.log 2>&1 &