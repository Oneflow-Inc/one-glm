# export ONEFLOW_DEBUG=1
bash scripts/ds_finetune_seq2seq.sh config_tasks/model_blocklm_base.sh config_tasks/seq_customization.sh  2>&1 | tee logs/test.log 


