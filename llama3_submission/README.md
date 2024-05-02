# LLaMA3 server

## Introduction
This is the LLaMA3 backend server for HELM evaluation, cloned from the [Neurips 2023 LLM Efficiency Challenge](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/tree/master/sample-submissions/llama_recipes). The background server is built on top of FastAPI and Huggingface Transformers.

## Setup
First setup the environment by running the following commands:
1. set your huggingface token for LLaMA3 8B model access `export HUGGINGFACE_TOKEN="[YOUR_HUGGINGFACE_TOKEN]"`
2. set your huggingface repo for model upload `export HUGGINGFACE_REPO="[YOUR_HUGGINGFACE_REPO]"`, if not set will use the default repo `llama3-8b`
3. install the requirements `pip install -r llama3_submission/fast_api_requirements.txt`
4. install the transformers, accelerate, datasets, and peft `pip install transformers accelerate datasets peft`
5. login to huggingface `huggingface-cli login --token $HUGGINGFACE_TOKEN`
6. start the server `bash start_server.sh`

Then the server will start on `0.0.0.0:8080`, and can be captured by HELM.

## API
- `/process`: process the input data and return the output(text/tokens/logprob/request_time).
- `/tokenize`: process the input data and return the tokenized output.
- `/decode`: process the input data and return the decoded output.