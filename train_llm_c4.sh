# --------------------------------------------- c4 -----------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python run_clm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --dataset_name c4 \
    --bf16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --batch_size 128 \
    --block_size 1024 \
    --lora_num_ranks 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_steps 50 \
    --save_total_limit 15 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --warmup_steps 5 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --output_dir loras/llama2-c4 \
