## First training (Lora)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --finetuning_type lora \
    --quantization_bit 4 \
    --template default \
    --dataset_dir data \
    --dataset openorca,alpaca_en,alpaca_gpt4_en,alpaca_cot \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 40 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --report_to none \
    --output_dir saves/LLaMA3-8B/lora/train_lora \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --plot_loss True

```

# Config history
- Because the server always crash

```bash
export TRANSFORMERS_CACHE=/workspace/anly5810/model_cache
export HF_HOME=/workspace/anly5810/model_cache
pip install --upgrade huggingface_hub
huggingface-cli login

cd /workspace/anly5810/llama_code/LLaMA-Factory
source ../llama_venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_PORT=5811
python src/train_web.py

```