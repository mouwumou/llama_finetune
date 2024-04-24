HUGGINGFACE_TOKEN="[YOUR_HUGGINGFACE_TOKEN]"

export TRANSFORMERS_CACHE=/workspace/model_cache
export HF_HOME=/workspace/model_cache

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]

pip install accelerate 
pip install --upgrade huggingface_hub
huggingface-cli login --token $HUGGINGFACE_TOKEN

export CUDA_VISIBLE_DEVICES=0
export GRADIO_SERVER_PORT=7860
python src/train_web.py