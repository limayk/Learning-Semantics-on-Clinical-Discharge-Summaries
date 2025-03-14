import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import os
from tqdm import tqdm
import gc
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Dataset class with chunking for long texts
class MedicalNotesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, chunk_strategy="first_chunk"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_strategy = chunk_strategy  # "first_chunk", "last_chunk", or "mean_pooling"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # For very long texts, we have several strategies
        if self.chunk_strategy == "first_chunk":
            # Just take the first chunk up to max_length
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        elif self.chunk_strategy == "last_chunk":
            # Take the last chunk (may contain conclusion/diagnosis)
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_length - 2:  # Account for special tokens
                tokens = tokens[-(self.max_length - 2):]
            
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
            
            # Pad if needed
            if len(token_ids) < self.max_length:
                padding_length = self.max_length - len(token_ids)
                token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
                
            attention_mask = [1] * len(token_ids)
            attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in token_ids]
            
            encoding = {
                'input_ids': torch.tensor([token_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Gradient accumulation training function for memory efficiency
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, 
                epochs=3, gradient_accumulation_steps=4, eval_every=1000):
    best_val_loss = float('inf')
    best_model_state = None
    global_step = 0
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        running_loss = 0
        train_preds, train_labels = [], []
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        optimizer.zero_grad()  # Zero gradients at the start of each epoch
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
            loss.backward()  # Backward pass
            
            train_loss += loss.item() * gradient_accumulation_steps
            running_loss += loss.item() * gradient_accumulation_steps
            
            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{running_loss / gradient_accumulation_steps:.4f}",
                    'global_step': global_step
                })
                running_loss = 0
                
                # Perform evaluation at regular intervals to save memory
                if global_step % eval_every == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    val_loss, val_metrics = evaluate_model(model, val_dataloader, device)
                    
                    logger.info(f"Step {global_step} - Validation: Loss: {val_loss:.4f}, "
                               f"Accuracy: {val_metrics['accuracy']:.4f}, "
                               f"F1: {val_metrics['f1']:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()
                        logger.info(f"Step {global_step} - New best model saved with val loss: {best_val_loss:.4f}")

                    # Set model back to training mode
                    model.train()
                    
                    # Free up memory
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Make sure to update with any remaining gradients at the end of epoch
        if (step + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        logger.info(f"Epoch {epoch + 1} - Avg training loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Full validation at the end of each epoch
        val_loss, val_metrics = evaluate_model(model, val_dataloader, device)
        logger.info(f"Epoch {epoch + 1} - Validation: Loss: {val_loss:.4f}, "
                   f"Accuracy: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"Epoch {epoch + 1} - New best model saved with val loss: {best_val_loss:.4f}")
        
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(dataloader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    cm = confusion_matrix(val_labels, val_preds)
    
    metrics = {
        'accuracy': val_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return avg_val_loss, metrics

# Function to load data in batches to avoid memory issues
def load_data_in_chunks(file_path, chunk_size=50000):
    data_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        data_chunks.append(chunk)
    return pd.concat(data_chunks)

# Main function to load data and train model
def main():
    # Configuration
    model_name = "emilyalsentzer/Bio_ClinicalBERT"  # Medical BERT model
    batch_size = 4  # Smaller batch size for longer texts
    epochs = 2
    learning_rate = 5e-6
    max_length = 512  # Max length for BERT
    gradient_accumulation_steps = 16  # Effective batch size = 8 × 8 = 64
    chunk_strategy = "first_chunk"  # Options: "first_chunk", "last_chunk"
    output_dir = "bertmodels"
    data_file = "medical_notes.csv"  # Replace with your file
    
    # Calculate training steps for learning rate scheduler
    total_samples = 179836  # Based on your dataset size 179836
    train_size = int(0.8 * total_samples)  # 80% for training
    steps_per_epoch = math.ceil(train_size / (batch_size * gradient_accumulation_steps))
    total_training_steps = steps_per_epoch * epochs
    
    logger.info(f"Total training steps: {total_training_steps}")
    
    # Load data in chunks to prevent memory issues
    try:
        data = pd.read_csv("notes/train_data.csv") #previous "data"
        texts = data['text'].values 
        labels = data['Y'].values
        print(labels.shape)
        print(texts.shape)
        print(set(labels))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Example for demonstration - replace with your actual data
        logger.warning("Using dummy data for demonstration.")
        # Create dummy data similar to your real data size
        texts = ["Patient presents with symptoms..." for _ in range(100)]  # Smaller for demo
        labels = np.random.randint(0, 2, size=100)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training on {len(train_texts)} examples, validating on {len(val_texts)} examples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Create datasets
    train_dataset = MedicalNotesDataset(train_texts, train_labels, tokenizer, max_length, 
                                         chunk_strategy=chunk_strategy)
    val_dataset = MedicalNotesDataset(val_texts, val_labels, tokenizer, max_length,
                                       chunk_strategy=chunk_strategy)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.15 * total_training_steps),  # 10% warmup
        num_training_steps=total_training_steps
    )
    
    # Move model to device
    model = model.to(device)
    
    # Train model
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_every=steps_per_epoch // 2  # Evaluate 4 times per epoch
    )
    
    # Save model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, "Trial_one")
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Function to evaluate on new data
    def predict(text):
        model.eval()
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]
            probability = probabilities[0][prediction].cpu().numpy()
        
        return prediction, probability
    
    # Example prediction
    example = "Patient shows signs of improvement after treatment."
    prediction, probability = predict(example)
    logger.info(f"Example text: '{example}'")
    logger.info(f"Prediction: {prediction} (class {prediction}) with probability {probability:.4f}")

if __name__ == "__main__":
    main()