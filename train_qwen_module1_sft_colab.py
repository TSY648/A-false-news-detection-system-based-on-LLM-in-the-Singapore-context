"""
Colab-friendly SFT / QLoRA training script for module-1 claim extraction.

Expected input dataset:
    module1_sft_1000.json

Each row should look like:
{
  "instruction": "Extract atomic factual claims, generate English search queries, and infer time ranges from the raw text. Return JSON only.",
  "input": {
    "today_date": "2026-01-07",
    "raw_text": "..."
  },
  "output": [
    {
      "claim": "...",
      "query": "...",
      "start_date": null,
      "end_date": null
    }
  ]
}

Recommended usage on Google Colab:

1. Runtime -> Change runtime type -> GPU
2. Install:
   !pip install -U transformers trl peft accelerate bitsandbytes datasets
3. Upload this file + module1_sft_1000.json
4. Run:
   !python train_qwen_module1_sft_colab.py \
       --train-file module1_sft_1000.json \
       --model-name Qwen/Qwen2.5-7B-Instruct \
       --output-dir qwen-module1-lora

Notes:
- Prefer LoRA / QLoRA on Colab.
- The output is a LoRA adapter, not a standalone merged model.
"""

import argparse
import json
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = "qwen-module1-lora"
DEFAULT_INSTRUCTION = (
    "Extract atomic factual claims, generate English search queries, "
    "and infer time ranges from the raw text. Return JSON only."
)


def load_sft_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Training file must be a JSON array.")
    return data


def build_prompt(record: Dict[str, Any]) -> str:
    instruction = str(record.get("instruction", "")).strip() or DEFAULT_INSTRUCTION
    input_payload = record.get("input", {})
    return (
        f"{instruction}\n\n"
        "Input JSON:\n"
        f"{json.dumps(input_payload, ensure_ascii=False, indent=2)}\n\n"
        "Output JSON:\n"
    )


def build_completion(record: Dict[str, Any]) -> str:
    output_payload = record.get("output", [])
    return json.dumps(output_payload, ensure_ascii=False)


def build_dataset(records: List[Dict[str, Any]]) -> Dataset:
    rows: List[Dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "prompt": build_prompt(record),
                "completion": build_completion(record),
            }
        )
    return Dataset.from_list(rows)


def build_quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def train(
    train_file: str,
    model_name: str,
    output_dir: str,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
) -> None:
    records = load_sft_records(train_file)
    train_dataset = build_dataset(records)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype="auto",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_seq_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        packing=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Output dir    : {output_dir}")
    print(f"Base model    : {model_name}")
    print("Saved LoRA adapter successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen module-1 claim extraction model with LoRA/QLoRA on Colab.")
    parser.add_argument("--train-file", required=True, help="Path to the SFT JSON file.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base Qwen model name.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save LoRA adapter.")
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        train_file=args.train_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
