import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def build_prompt(example):
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    if ctx:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{ctx}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    resp = (example.get("response") or "").strip()
    # SFT: 训练目标是 prompt + response（teacher forcing）
    return {"text": prompt + resp}

def main():
    student_id = "gpt2"

    # GPT-2 context limit
    MODEL_MAX_CTX = 1024
    # SFT 阶段不需要 rollout completion，但仍要避免超长
    MAX_SFT_TOKENS = 1024

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.map(build_prompt, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]

    tok = AutoTokenizer.from_pretrained(student_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(ex):
        out = tok(
            ex["text"],
            truncation=True,
            max_length=MAX_SFT_TOKENS,
            padding=False,
        )
        return out

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(student_id)

    # Causal LM collator
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir="phase1_sft_gpt2_dolly",
        num_train_epochs=3,                 # 论文：3 epochs
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,                 # 你可以按论文 baseline 网格搜索；先跑通可先用 5e-5
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,        # 关键：按 eval_loss 选最好 checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model("phase1_sft_gpt2_dolly/best")
    tok.save_pretrained("phase1_sft_gpt2_dolly/best")

if __name__ == "__main__":
    main()
