# HELM

## Introduction
This is the HELM configuration and running script for testing, cloned from the [Neurips 2023 LLM Efficiency Challenge](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge).

## How to install HELM
Install HELM: `pip install git+https://github.com/stanford-crfm/helm.git`

## Build a HTTP server
Please see llama3_submission README for building a HTTP server.

## Configure HELM
use `python build_run_specs_full.py --example_budget <budget_number>` to generate a configuration file with specific budget_number.

## Run HELM
```bash
helm-run --conf-paths run_specs_full_coarse_600_budget.conf --suite v1 --max-eval-instances 10
helm-summarize --suite v1
```
`helm-run` will run the experiments and `helm-summarize` will summarize the results.

## See HELM Results
```bash
helm-server
```

## Evaluation
base on the challenge rule, the score will be calculated by:
$$
score = \prod(\text{mean-win-rate}(task))
$$