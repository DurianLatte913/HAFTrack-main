SCRIPT=haftrack
CONFIG=vit_cte_haf_lasher
nohup python tracking/test.py \
    ${SCRIPT} \
    ${CONFIG} \
    --dataset RGBT234 \
    --threads 32 \
    --num_gpus 4 \
    --vis_gpus 0,1,2,3 \
    >./experiments/${SCRIPT}/test_${SCRIPT}-${CONFIG}.log 2>&1 &