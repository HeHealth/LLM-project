import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging

# Configuration
DATA_DIR = "/home/yzhong43/redTeam/processed_data"
OUTPUT_DIR = "/home/yzhong43/redTeam/preference_model"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
BATCH_SIZE = 4  # Reduced to avoid OOM
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 4 * 2 = 8
LEARNING_RATE = 1e-5
EPOCHS = 3
MAX_LENGTH = 512
NUM_WORKERS = 2  # Reduced to minimize memory overhead
LOG_FILE = "/home/yzhong43/redTeam/preference_model_training.log"

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

class PreferenceDataset(Dataset):
    """Dataset for preference model training."""
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON in {file_path}: {e}")
            logger.info(f"Loaded {len(self.data)} items from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        chosen = item['chosen']
        rejected = item['rejected']
        # Format inputs
        chosen_input = f"[CONTEXT]{context}[RESPONSE]{chosen}[/RESPONSE]"
        rejected_input = f"[CONTEXT]{context}[RESPONSE]{rejected}[/RESPONSE]"
        try:
            chosen_tokens = self.tokenizer(
                chosen_input,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            rejected_tokens = self.tokenizer(
                rejected_input,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
        except Exception as e:
            logger.error(f"Error tokenizing item {idx}: {e}")
            raise
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze()
        }

def compute_loss(model, chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask):
    """Compute pairwise ranking loss."""
    with autocast():
        try:
            chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
            chosen_scores = chosen_outputs.logits[:, -1].mean(dim=-1)
            rejected_scores = rejected_outputs.logits[:, -1].mean(dim=-1)
            loss = -torch.log(torch.sigmoid(chosen_scores - rejected_scores)).mean()
            return loss
        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            raise

def evaluate(model, dataloader, device):
    """Evaluate ranking accuracy on validation set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            try:
                chosen_input_ids = batch['chosen_input_ids'].to(device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(device)
                rejected_input_ids = batch['rejected_input_ids'].to(device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(device)
                with autocast():
                    chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
                    rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
                    chosen_scores = chosen_outputs.logits[:, -1].mean(dim=-1)
                    rejected_scores = rejected_outputs.logits[:, -1].mean(dim=-1)
                    correct += (chosen_scores > rejected_scores).sum().item()
                    total += chosen_scores.size(0)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                continue
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    """Train the preference model."""
    # Initialize tokenizer and model with 4-bit quantization
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
        # Set padding token to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            token=os.environ.get("HF_TOKEN")
        )
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        raise

    # Apply LoRA
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA applied successfully")
    except Exception as e:
        logger.error(f"Error applying LoRA: {e}")
        raise

    # Load datasets
    try:
        train_dataset = PreferenceDataset(os.path.join(DATA_DIR, "preference_train.jsonl"), tokenizer, MAX_LENGTH)
        val_dataset = PreferenceDataset(os.path.join(DATA_DIR, "preference_val.jsonl"), tokenizer, MAX_LENGTH)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        logger.info(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation items")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

    # Initialize optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        step_count = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader, 1):
            try:
                chosen_input_ids = batch['chosen_input_ids'].to(device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(device)
                rejected_input_ids = batch['rejected_input_ids'].to(device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(device)

                loss = compute_loss(model, chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()

                if step % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                step_count += 1
                if step % (100 * GRADIENT_ACCUMULATION_STEPS) == 0:
                    logger.info(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                torch.cuda.empty_cache()  # Clear memory on error
                continue

        avg_loss = total_loss / step_count if step_count > 0 else 0
        val_accuracy = evaluate(model, val_dataloader, device)
        logger.info(f"Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save model
    try:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Model saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    main()