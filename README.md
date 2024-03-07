# LLM LoRA fine-tuning

## Finetune Setup

```bash
conda create --name llama2 python=3.9
conda activate llama2

# pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# transformers
pip install git+https://github.com/huggingface/transformers@b074461ef0f54ce37c5239d30ee960ece28d11ec

# flash attention
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Remaining
pip install -r requirements.txt

# Note: If you encounter problem that is about loading datasets, try upgrade datasets package first.
pip install -U datasets
```

## Fine-tuning

* Single GPU training

```
./train_llm.sh      # Use wikitext2
./train_llm_c4.sh   # Use c4
```

* Multi-GPU trainig (DDP)

```
./train_llm_multigpu.sh
./train_llm_c4_multigpu.sh
```

## Batch size config

You may need to adjust batch size due to memory constraint.

Modify the arguments in `train_llm.sh` or `train_llm_c4.sh`.

```
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
```