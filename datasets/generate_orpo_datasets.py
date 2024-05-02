import hashlib
import json

from datasets import load_dataset


if __name__ == "__main__":
    dataset = []
    raw_dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")
    raw_dataset = raw_dataset.map(lambda r: {"length": len(r["chosen"])})
    for example in raw_dataset:
        history = []
        for i in range((example["length"] - 2) // 2):
            history.append(
                [
                    example["chosen"][2 * i]["content"].strip(),
                    example["chosen"][2 * i + 1]["content"].strip(),
                ]
            )

        dataset.append(
            {
                "prompt": example["chosen"][-2]["content"].strip(),
                "answer": [
                    example["chosen"][-1]["content"].strip(),
                    example["rejected"][-1]["content"].strip(),
                ],
                "history": history,
            }
        )

    with open("orpo_data.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)