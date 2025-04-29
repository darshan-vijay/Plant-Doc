# --- Fine-tuning Flan-T5-Large with LoRA for Plant Disease QA (Kaggle Version) ---
import os
import torch
import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# --- Load datasets ---
with open('/kaggle/input/qa-diseases/QA_diseases.json') as f:
    qa_data = json.load(f)

with open('/kaggle/input/qa-diseases/disease_context.json') as f:
    disease_contexts = json.load(f)

# --- Prepare dataset ---
inputs, targets = [], []
for item in qa_data:
    disease = item['disease']
    context = disease_contexts[disease]
    question = item['input']
    answer = item['target']
    input_text = f"context: {context} question: {question}"
    inputs.append(input_text)
    targets.append(answer)

dataset = Dataset.from_dict({"input": inputs, "target": targets})

# --- Set up device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Load tokenizer and model ---
model_checkpoint = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# --- Configure LoRA ---
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
    bias="none",
)

# --- Apply LoRA ---
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(device)

# --- Preprocessing ---
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['input'],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        examples['target'],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# --- Train/Val Split ---
train_test = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test['train']
eval_dataset = train_test['test']

print(f"Total: {len(tokenized_dataset)}, Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# --- TrainingArguments ---
training_args = TrainingArguments(
    output_dir="/kaggle/working/lora-flan-t5-large-plant-qa",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=25,
    per_device_train_batch_size=2,  # 2 because Kaggle GPUs have memory limits
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

import numpy as np
import evaluate

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if hasattr(predictions, "shape") and len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)

    if hasattr(predictions, 'tolist'):
        predictions = predictions.tolist()
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()

    processed_labels = []
    for label in labels:
        processed_label = [l if l != -100 else tokenizer.pad_token_id for l in label]
        processed_labels.append(processed_label)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(processed_labels, skip_special_tokens=True)

    # Compute ROUGE
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {key: value * 100 for key, value in rouge_result.items()}

    # Compute BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["bleu"] * 100

    # Combine
    metrics = {
        "rouge1": rouge_result.get("rouge1", 0),
        "rouge2": rouge_result.get("rouge2", 0),
        "rougeL": rouge_result.get("rougeL", 0),
        "bleu": bleu_score
    }

    return metrics



# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- Start Training ---
print("Starting training...")
trainer.train()

# --- Save Final LoRA Adapter ---
model.save_pretrained("/kaggle/working/lora-flan-t5-large-plant-qa/final")
print("✅ Fine-tuning complete. Model saved to /kaggle/working/lora-flan-t5-large-plant-qa/final")

# --- Test Model ---
print("\nTesting model with a sample question...")
sample_input = dataset[0]['input']
sample_target = dataset[0]['target']

inputs = tokenizer(sample_input, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=50)
prediction = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Input: {sample_input[:100]}...")
print(f"Target: {sample_target}")
print(f"Prediction: {prediction}")
