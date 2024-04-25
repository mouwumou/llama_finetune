HUGGINGFACE_TOKEN="[YOUR_HUGGINGFACE_TOKEN]"

export TRANSFORMERS_CACHE=/workspace/model_cache
export HF_HOME=/workspace/model_cache

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]

pip install accelerate 
pip install --upgrade huggingface_hub
huggingface-cli login --token $HUGGINGFACE_TOKEN

pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_PORT=7860
python src/train_web.py