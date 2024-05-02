# LLaMA3 server

## Introduction
This is the LLaMA3 backend server for HELM evaluation, cloned from the [Neurips 2023 LLM Efficiency Challenge](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/tree/master/sample-submissions/llama_recipes). The background server is built on top of FastAPI and Huggingface Transformers.

## Setup
```bash
HUGGINGFACE_TOKEN="[YOUR_HUGGINGFACE_TOKEN]"

# export TRANSFORMERS_CACHE=/workspace/model_cache # optional if you want to cache models
# export HF_HOME=/workspace/model_cache # optional if you want to cache models

pip install -r llama3_submission/fast_api_requirements.txt
pip install transformers accelerate datasets peft
huggingface-cli login --token $HUGGINGFACE_TOKEN

bash start_server.sh
```
Then the server will start on `0.0.0.0:8080`, and can be captured by HELM.

## API
- `/process`: process the input data and return the output(text/tokens/logprob/request_time).
- `/tokenize`: process the input data and return the tokenized output.
- `/decode`: process the input data and return the decoded output.