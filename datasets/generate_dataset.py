from datasets import load_dataset
# from transformers import AutoTokenizer, LlamaForCausalLM
import json
import os
import io
import random
# import tqdm
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode,encoding='utf-8')
    return f
#
def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default,ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f
def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
result_dataset = []


mmlu_data = load_dataset('cais/mmlu', name='all', split="auxiliary_train", cache_dir='./dataset_dir')

for item in mmlu_data:
    insturction='Question: This question refers to the following information.\n'
    choices=''
    for idx,choice in enumerate(item['choices']):
        choices+=chr(ord('A')+idx)+'.'+choice+'\n'
    choices+='Answer:'
    result_dataset.append({
        'instruction':insturction,
        'input':item['question']+'\n'+choices,
        'output':chr(ord('A')+item['answer']),
        'data_source':'mmlu'
    })
result_dataset = random.sample(result_dataset,int(len(result_dataset)/10))
#
bigbench = []
bigbench_analytic_entailment = load_dataset('tasksource/bigbench',name = 'analytic_entailment', cache_dir='./dataset_dir')
bigbench_causal_judgment = load_dataset('tasksource/bigbench',name = 'causal_judgment', cache_dir='./dataset_dir')
bigbench_emoji_movie = load_dataset('tasksource/bigbench',name = 'emoji_movie', cache_dir='./dataset_dir')
bigbench_empirical_judgments = load_dataset('tasksource/bigbench',name = 'empirical_judgments', cache_dir='./dataset_dir')
bigbench_known_unknowns = load_dataset('tasksource/bigbench',name = 'known_unknowns', cache_dir='./dataset_dir')
bigbench_logical_deduction = load_dataset('tasksource/bigbench',name = 'logical_deduction', cache_dir='./dataset_dir')
bigbench_strange_stories = load_dataset('tasksource/bigbench',name = 'strange_stories', cache_dir='./dataset_dir')
bigbench_snarks = load_dataset('tasksource/bigbench',name = 'snarks', cache_dir='./dataset_dir')
bigbench_dark_humor_detection = load_dataset('tasksource/bigbench',name = 'dark_humor_detection', cache_dir='./dataset_dir')
bigbench.append(bigbench_analytic_entailment)
bigbench.append(bigbench_causal_judgment)
bigbench.append(bigbench_emoji_movie)
bigbench.append(bigbench_empirical_judgments)
bigbench.append(bigbench_known_unknowns)
bigbench.append(bigbench_logical_deduction)
bigbench.append(bigbench_strange_stories)
bigbench.append(bigbench_snarks)
bigbench.append(bigbench_dark_humor_detection)

#处理bigbench数据的格式
for dataset in bigbench:
    if len(dataset)>3:
        for item in dataset:
            insturction = 'Question: \n'
            choices = ''
            for idx, choice in enumerate(item['multiple_choice_targets']):
                choices += chr(ord('A') + idx) + '.' + choice + '\n'
            choices += 'Answer:'
            result_dataset.append({
                'instruction': insturction,
                'input': item['inputs'] + '\n' + choices,
                'output': chr(ord('A') + item['multiple_choice_targets'].index(item['targets'][0])),
                'data_source': 'bigbench'
            })
    else:
        for index in dataset:
            for item in dataset[index]:
                insturction = 'Question: \n'
                choices = ''
                for idx, choice in enumerate(item['multiple_choice_targets']):
                    choices += chr(ord('A') + idx) + '.' + choice + '\n'
                choices += 'Answer:'
                result_dataset.append({
                    'instruction': insturction,
                    'input': item['inputs'] + '\n' + choices,
                    'output': chr(ord('A') + item['multiple_choice_targets'].index(item['targets'][0])),
                    'data_source': 'bigbench'
                })
print(len(result_dataset))


truthful_qa = load_dataset('truthful_qa',name = 'multiple_choice', cache_dir='./dataset_dir')
if len(truthful_qa)>3:
    for item in truthful_qa:
        insturction = 'Question: \n'
        choices = ''
        for idx, choice in enumerate(item['mc1_targets']['choices']):
            choices += chr(ord('A') + idx) + '.' + choice + '\n'
        choices += 'Answer:'
        result_dataset.append({
            'instruction': insturction,
            'input': item['question'] + '\n' + choices,
            'output': chr(ord('A') + item['mc1_targets']['labels'].index(1)),
            'data_source': 'truthful_qa'
        })
else:
    for index in truthful_qa:
        for item in truthful_qa[index]:
            insturction = 'Question: \n'
            choices = ''
            for idx, choice in enumerate(item['mc1_targets']['choices']):
                choices += chr(ord('A') + idx) + '.' + choice + '\n'
            choices += 'Answer:'
            result_dataset.append({
                'instruction': insturction,
                'input': item['question'] + '\n' + choices,
                'output': chr(ord('A') + item['mc1_targets']['labels'].index(1)),
                'data_source': 'truthful_qa'
            })

print(len(result_dataset))
temp_result_dataset = []
dailymail = load_dataset('cnn_dailymail',name = '3.0.0',split='train', cache_dir='./dataset_dir')
for item in dailymail:
    insturction = '	### Article: \n'
    temp_result_dataset.append({
        'instruction': insturction,
        'input': item['article'] + '\n',
        'output': 'Summarize the above article in 3 sentences.' + item['highlights'],
        'data_source': 'dailymail'
    })
temp_result_dataset = random.sample(temp_result_dataset,int(0.1*len(temp_result_dataset)))
result_dataset.extend(temp_result_dataset)

print(len(result_dataset))
gsm = load_dataset('gsm8k',name='main',split='train', cache_dir='./dataset_dir')
if len(gsm)<=3:
    for index in gsm:
        for item in gsm[index]:
            insturction = 'Q: \n'
            result_dataset.append({
                'instruction': insturction,
                'input': item['question'] + '\n',
                'output': 'A:' + item['answer'],
                'data_source': 'gsm'
            })
else:
    for item in gsm:
        insturction = 'Q: \n'
        result_dataset.append({
            'instruction': insturction,
            'input': item['question'] + '\n',
            'output': 'A:' + item['answer'],
            'data_source': 'gsm'
        })

print(len(result_dataset))

bbq = load_dataset('lighteval/bbq_helm', name='all', split='test', cache_dir='./dataset_dir')
for item in bbq:
    insturction = 'The following are multiple choice questions (with answers). \n'
    choices = ''
    for idx, choice in enumerate(item['choices']):
        choices += chr(ord('A') + idx) + '.' + choice + '\n'
    choices += 'Answer:'
    result_dataset.append({
        'instruction': insturction,
        'input': 'Passage:' + item['context'] + '\n' + item['question'] + '\n' + choices,
        'output': chr(ord('A') + item['gold_index']),
        'data_source': 'bbq'
    })

print(len(result_dataset))

flan_data = load_dataset('Muennighoff/flan', split="train", cache_dir='./dataset_dir')

for item in flan_data:
    insturction = ''
    result_dataset.append({
        'instruction': insturction,
        'input': item['inputs'] + '\n',
        'output': item['targets'],
        'data_source': 'flan'
    })

print(len(result_dataset))

databricks_dolly = load_dataset('databricks/databricks-dolly-15k', cache_dir='./dataset_dir')
if len(databricks_dolly)>3:
    for item in databricks_dolly:
        insturction = item['instruction'] + '\n'
        result_dataset.append({
            'instruction': insturction,
            'input': item['context'] + '\n',
            'output': item['response'],
            'data_source': 'databricks_dolly'
        })
else:
    for index in databricks_dolly:
        for item in databricks_dolly[index]:
            insturction = item['instruction'] + '\n'
            result_dataset.append({
                'instruction': insturction,
                'input': item['context'] + '\n',
                'output': item['response'],
                'data_source': 'databricks_dolly'
            })
print(len(result_dataset))


random.shuffle(result_dataset)
result_dataset = random.sample(result_dataset,int(len(result_dataset)*0.67))
jdump(result_dataset,'../LLaMA-Effcient-Tuning/data_for_fintune/nips_data_add_v7.json')
print(len(result_dataset))