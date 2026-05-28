"""
Training script for math AI model.

Fine-tunes a pre-trained math model (Qwen2.5-Math) using LoRA
for parameter-efficient fine-tuning on consumer GPUs.
"""

import os
import yaml
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def load_config(config_path="configs/training_config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_conversations(example, tokenizer):
    """Convert conversation format to a single text string."""
    messages = example["conversations"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def setup_model_and_tokenizer(model_config):
    """Load model with quantization and tokenizer."""
    model_name = model_config["name"]
    print(f"Loading model: {model_name}")

    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(model, lora_config):
    """Configure and apply LoRA adapters."""
    print("Setting up LoRA...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


def main():
    config = load_config()

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk("data/math_dataset")
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config["model"])
    model = setup_lora(model, config["lora"])

    # Format conversations to text
    print("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_conversations(x, tokenizer),
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    train_config = config["training"]
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        eval_steps=train_config["eval_steps"],
        save_total_limit=train_config["save_total_limit"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        optim=train_config["optim"],
        max_grad_norm=train_config["max_grad_norm"],
        report_to=train_config["report_to"],
        max_seq_length=config["model"]["max_seq_length"],
        dataset_text_field="text",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the final model
    print("\nSaving model...")
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")

    # Save training metrics
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
