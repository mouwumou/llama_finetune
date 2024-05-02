# LLaMA3 Train

## Introduction
This is the LLaMA3 training script, our training is based on the Finetuning Framework [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). This fine-tuning library encapsulates enough training and acceleration methods, making it very suitable as a support for our training and debugging.

## Build LLaMA Factory Environment
```bash
HUGGINGFACE_TOKEN="[YOUR_HUGGINGFACE_TOKEN]"

# export TRANSFORMERS_CACHE=/workspace/model_cache # optional if you want to cache models
# export HF_HOME=/workspace/model_cache # optional if you want to cache models

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]

pip install accelerate 
pip install --upgrade huggingface_hub
huggingface-cli login --token $HUGGINGFACE_TOKEN
```
Then you can start the training script by running the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --finetuning_type lora \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset merged_data,lima \ 
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 20 \
    --warmup_steps 10 \
    --optim adamw_torch \
    --report_to none \
    --output_dir saves/LLaMA3-8B/lora/merged_lora_second \
    --bf16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target q_proj,v_proj \
    --plot_loss True
```
- Note: Remember to put `merged_data` in the `data` folder, and add info in `data/dataset_info.json`.
```json
{
 "merged_data": {
    "file_name": "merged_data.json",
  }
}
```

and for ORPO dataset, you can use the following info:
```json
{
  "orpo_data": {
    "file_name": "orpo_data.json",
    "ranking": true,
    "columns": {
      "prompt": "prompt",
      "response": "answer",
      "history": "history"
    }
  }
}
```

## Accelerate
There are also some acceleration methods in the LLaMA-Factory, you can install them by running the following command:
```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```
- Remember these libraries need dependend CUDA versions.

## Train Web
For single GPU training, you can use the following command to start the web server:
```bash
export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_PORT=7860
python src/train_web.py

```

