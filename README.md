# llama_finetune
## Introduction
This is finetune experiment on LLaMa3 for course DSAN5810. This respotory includes mainly 4 parts:
1. datasets generation
2. llama finetune training
3. HELM
4. LLaMA3 backend server

Please refer to the README.md in each folder for more details.

## File structure
```bash
.
├── datasets  # datasets generation
│   ├── README.md
│   ├── generate_datasets.py
├── helm  # HELM evaluation
│   ├── README.md
│   ├── helm_commands.sh
│   ├── build_run_specs_full.py
│   ├── run_specs_full_coarse_600_budget.conf
├── llama_train  # llama finetune training
│   ├── README.md
├── llama3_submission  # LLaMA3 backend server
│   ├── api.py
│   ├── Dockerfile
│   ├── fast_api_requirements.txt
│   ├── main.py
│   └── start_server.sh
├── presentation  # presentation slides
│   ├── imgs
│   ├── DSAN5810 Final Project.pdf
│   ├── DSAN5810 Final Project.pptx
├── README.md
```

