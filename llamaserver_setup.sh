HUGGINGFACE_TOKEN="[YOUR_HUGGINGFACE_TOKEN]"

export TRANSFORMERS_CACHE=/workspace/model_cache
export HF_HOME=/workspace/model_cache

pip install -r llama3_submission/fast_api_requirements.txt
pip install transformers accelerate datasets
huggingface-cli login --token $HUGGINGFACE_TOKEN

cd llama3_submission

bash start_server.sh