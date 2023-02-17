# export ONEFLOW_DEBUG=1

bash scripts/ds_finetune_seq2seq.sh config_tasks/model_blocklm_base.sh config_tasks/seq_cnndm_org.sh  2>&1 | tee logs/test.log 


# python finetune_glm.py --finetune --experiment-name _02-17-10-10 --task --data-dir --save /home/fengwen/one-glm/runs --checkpoint-activations --num-workers 1 --no-load-lr-scheduler --model-parallel-size 1 --overwrite 2>&1 | tee logs/log-_02-17-10-10.txt