DATA_ROOT=/data/dataset/fengwen/resume-zip/other/dataset/  #/dataset/fd5061f6/tuteng/BlockLM/data
CHECKPOINT_PATH=/home/fengwen/datasets/CHECKPOINT_PATH # /data/dataset/fengwen/resume-zip/other/checkpoints/blocklm-base-blank
SAVE_PATH=/home/fengwen/GLM/datasets/save_path
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --master_port $MASTER_PORT --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"

EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
mkdir logs
run_cmd="${DISTRIBUTED_ARGS} finetune_glm.py \
       --deepspeed \
       --deepspeed_config config_tasks/config_blocklm_10B.json \
       --finetune \
       --cloze-eval \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 2 \
       --save-epoch 100000 \
       --num-workers 1 \
       --no-load-optim \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --pattern-id 0 \
       --model-parallel-size ${MP_SIZE} \
       --epochs ${XXLARGE_EPOCH} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt"
       # --fp16 \

echo ${run_cmd}
eval ${run_cmd}
