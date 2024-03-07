# ------------------------------- wikitext2 ------------------------------------

CUDA_VISIBLE_DEVICES='0,1' torchrun --nproc_per_node 2 run_clm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --bf16 \
    --batch_size 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 6 \
    --gradient_checkpointing \
    --block_size 1024 \
    --lora_num_ranks 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --save_steps 10 \
    --save_total_limit 15 \
    --do_train \
    --do_eval \
    --output_dir loras/llama2-w2 \
    --report_to "none"


