import os
import torch
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Config
MODEL_PATH = "/data/models_weights/phi-2-legal-finetuned"  # Your local phi-2 weights
DATA_PATH = "legal_instructions.jsonl"
OUTPUT_DIR = "phi2-legal-finetuned"
USE_LORA = True
BF16 = torch.cuda.is_bf16_supported()

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Load model and tokenizer with memory optimization
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    device_map="auto",
    load_in_8bit=True  # Use 8-bit quantization to reduce memory usage
)

# Apply LoRA
if USE_LORA:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,  # Reduced from 16 to save memory
        lora_alpha=16,  # Reduced from 32 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Print trainable params to confirm LoRA setup

# Load dataset
def load_instruction_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l.strip()) for l in f.readlines()]
    return Dataset.from_list(lines)

dataset = load_instruction_dataset(DATA_PATH)

# Format dataset for causal LM with reduced sequence length
def format_example(example):
    prompt = f"""### Instruction:
{example['instruction']}
### Context:
{example['context']}
### Response:
{example['response']}"""
    return {"input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"].squeeze()}

dataset = dataset.map(format_example)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Increased from 4
    gradient_checkpointing=True,  # Enable gradient checkpointing
    save_strategy="epoch",
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=not BF16,
    bf16=BF16,
    logging_dir="./logs",
    logging_steps=20,
    save_total_limit=2,
    report_to="none",
    overwrite_output_dir=True,
    optim="adamw_torch_fused",  # Use memory-efficient optimizer
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start fine-tuning
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)