#!/usr/bin/env python3
"""Build few-shot GSM8K prompts using standard 5 examples from the training set.
Usage:
  python tools/build_fewshot_prompts.py -o prompts/reasoning_5shot.json
"""
import json, re, argparse
from pathlib import Path
from datasets import load_dataset

FEWSHOT_TEMPLATE = """Q: {question}
A: {answer}

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='prompts/reasoning_5shot.json')
    parser.add_argument('--n-shot', type=int, default=5)
    args = parser.parse_args()

    # Load GSM8K train (for few-shot examples) and test (for questions)
    train = load_dataset('openai/gsm8k', 'main', split='train')
    test = load_dataset('openai/gsm8k', 'main', split='test')

    # Pick first N examples from train as few-shot
    fewshot_examples = []
    for i in range(args.n_shot):
        fewshot_examples.append({
            'question': train[i]['question'],
            'answer': train[i]['answer'],
        })

    # Build few-shot prefix
    prefix = ''
    for ex in fewshot_examples:
        prefix += FEWSHOT_TEMPLATE.format(question=ex['question'], answer=ex['answer'])

    # Build test prompts with few-shot prefix
    prompts = []
    for item in test:
        full_prompt = prefix + f"Q: {item['question']}\nA:"
        prompts.append({
            'question': full_prompt,
            'answer': item['answer'],
        })

    out_path = Path(__file__).parent.parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"Built {len(prompts)} {args.n_shot}-shot prompts")
    print(f"Prefix length: {len(prefix)} chars")
    print(f"First prompt preview: {prompts[0]['question'][:200]}...")
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
