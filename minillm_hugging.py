import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.experimental.minillm import MiniLLMTrainer, MiniLLMConfig


# --------- 1) Prompt template ----------
def format_prompt(example):
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    if ctx:
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{ctx}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    return {"prompt": prompt, "reference": (example.get("response") or "").strip()}


def main():
    # ========= Phase 1 -> Phase 2: student init from SFT checkpoint =========
    # 改成你 Phase 1 输出的 best checkpoint 路径
    # 例如：phase1_sft_gpt2_dolly/best
    phase1_student_dir = "phase1_sft_gpt2_dolly/best"

    teacher_id = "gpt2-large"

    # ========= Paper-aligned Phase 2 settings =========
    # Paper: max length = 512, max completion length in TRL config is usually completion tokens
    MODEL_MAX_CTX = 512
    MAX_NEW_TOKENS = 128
    MAX_PROMPT_TOKENS = MODEL_MAX_CTX - MAX_NEW_TOKENS  # 384

    # ========= Load dataset =========
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.map(format_prompt, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]

    # ========= Tokenizer (load from Phase 1 dir to ensure exact match) =========
    tokenizer = AutoTokenizer.from_pretrained(phase1_student_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========= Truncate prompts to avoid exceeding context =========
    def truncate_prompt(ex):
        ids = tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )["input_ids"]
        ex["prompt"] = tokenizer.decode(ids, skip_special_tokens=True)
        return ex

    train_ds = train_ds.map(truncate_prompt)
    eval_ds = eval_ds.map(truncate_prompt)

    def keep_ok(ex):
        if not ex["prompt"].strip():
            return False
        if not ex["reference"].strip():
            return False
        n = len(tokenizer(ex["prompt"])["input_ids"])
        return 1 <= n <= MAX_PROMPT_TOKENS

    train_ds = train_ds.filter(keep_ok)
    eval_ds = eval_ds.filter(keep_ok)

    # ========= Models =========
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Student: load from Phase 1 checkpoint (关键改动)
    student = AutoModelForCausalLM.from_pretrained(phase1_student_dir).to(device)

    # Teacher: same device, frozen
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ========= MiniLLMConfig (paper-aligned where possible) =========
    args = MiniLLMConfig(
        output_dir="phase2_minillm_gpt2_dolly",

        # Paper: lr=5e-6
        learning_rate=5e-6,

        # Paper: train for 5000 steps
        max_steps=5000,

        # Paper: mini-batch size 64
        # 你单卡 per_device_train_batch_size=2，则用 grad_accum=32 近似达到有效 batch=64
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,

        logging_steps=10,
        save_steps=500,
        save_total_limit=2,

        # Generation (paper: temperature=1)
        max_completion_length=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=1.0,

        # MiniLLM core (keep as you had)
        rkl_advantage=True,
        single_step_decomposition=True,
        length_normalization=True,
        gamma=0.0,

        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # 备注：论文的 PPO clipping epsilon=0.2、一次收集256句、4 inner epochs
    # 在 TRL 版 MiniLLMTrainer 里这部分可能被封装为内部默认值或其它字段名。
    # 如果你的 MiniLLMConfig 支持相关字段，你可以在这里追加：
    #   cliprange=0.2 或 epsilon=0.2
    #   num_generations=256
    #   num_inner_epochs=4
    # 但字段名依版本而异，先保证可运行。

    trainer = MiniLLMTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
