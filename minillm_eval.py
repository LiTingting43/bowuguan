import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def format_prompt(example):
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    if ctx:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{ctx}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    return {"prompt": prompt, "reference": (example.get("response") or "").strip()}

@torch.no_grad()
def main():
    model_dir = "gpt2-large"   # 训练输出目录
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # eval set：与训练脚本一致
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.map(format_prompt, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    eval_ds = ds["test"]

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    model.eval()

    rouge = evaluate.load("rouge")

    preds, refs = [], []
    max_new_tokens = 128

    for ex in eval_ds.select(range(min(500, len(eval_ds)))):  # 先评估 500 条，跑得快
        prompt = ex["prompt"]
        ref = ex["reference"]

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)

        # 只取 Response 之后的内容（避免 prompt 干扰）
        pred = text.split("### Response:\n", 1)[-1].strip()

        preds.append(pred)
        refs.append(ref)

    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

    # 你要的 “RL分数” 通常对应 rougeL（或 rougeLsum）
    print("ROUGE scores:", scores)
    print("R-L (rougeL):", scores["rougeL"])

if __name__ == "__main__":
    main()
