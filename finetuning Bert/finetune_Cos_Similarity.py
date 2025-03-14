import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import os
import signal
import sys
from tqdm import tqdm
import gc
import math
import time
import re

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

# Global variable to track manual stopping
STOP_TRAINING = False

# Signal handler for manual stopping
def signal_handler(sig, frame):
    global STOP_TRAINING
    logger.info("Manual stop signal received. Will stop after current batch...")
    STOP_TRAINING = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Custom model that outputs embeddings
class BertEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super(BertEmbeddingModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token embedding (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Normalize embedding to unit length for cosine similarity
        normalized_embedding = F.normalize(cls_output, p=2, dim=1)
        
        return normalized_embedding

# Cosine similarity loss
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        # Create a matrix of all pairs of embeddings
        batch_size = embeddings.size(0)
        
        # Calculate cosine similarity matrix
        cos_sim = torch.mm(embeddings, embeddings.t())
        
        # Create binary mask for positive pairs (same class)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~pos_mask
        
        # Exclude self-comparisons
        pos_mask.fill_diagonal_(False)
        neg_mask.fill_diagonal_(False)
        
        # Calculate loss: maximize similarity for same class, minimize for different class
        pos_loss = 1 - cos_sim[pos_mask]
        neg_loss = F.relu(cos_sim[neg_mask] - self.margin)
        
        # Average the losses
        pos_loss = pos_loss.mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        neg_loss = neg_loss.mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        
        return pos_loss + neg_loss

# Hierarchical chunking dataset for long texts
class HierarchicalMedicalNotesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, overlap=100, max_chunks=8):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.max_chunks = max_chunks  # Maximum number of chunks per document
    
    def __len__(self):
        return len(self.texts)
    
    def chunk_text(self, text):
        # If text is shorter than max_length, just use it directly
        if len(self.tokenizer.tokenize(text)) <= self.max_length:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            chunks = [{
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }]
            return chunks
        
        # For longer texts, create overlapping chunks that respect sentence boundaries
        chunks = []
        
        # Pattern for finding sentence boundaries (., !, ? followed by space)
        sentence_delimiters = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            # Get token count for this sentence
            sentence_tokens = len(self.tokenizer.tokenize(sentence))
            
            # If a single sentence is longer than max_length, we'll need to truncate it
            if sentence_tokens > self.max_length:
                # If we have content in the current chunk, finalize it first
                if current_tokens > 0:
                    encoding = self.tokenizer(
                        current_chunk,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    chunks.append({
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0)
                    })
                
                # Now handle the long sentence (we'll need to truncate it)
                # Tokenize the sentence
                long_tokens = self.tokenizer.tokenize(sentence)
                stride = self.max_length - self.overlap
                
                for i in range(0, len(long_tokens), stride):
                    chunk_tokens = long_tokens[i:i + self.max_length]
                    if len(chunk_tokens) < 100:  # Skip very small chunks
                        continue
                        
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                    encoding = self.tokenizer(
                        chunk_text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    chunks.append({
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0)
                    })
                
                # Reset the current chunk
                current_chunk = ""
                current_tokens = 0
                
            # If adding this sentence would exceed max_length, finalize current chunk and start a new one
            elif current_tokens + sentence_tokens > self.max_length:
                # Finalize current chunk
                encoding = self.tokenizer(
                    current_chunk,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                chunks.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0)
                })
                
                # Start new chunk with this sentence
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                if current_chunk:  # Add space if not the first sentence in chunk
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk if there's anything left
        if current_chunk and current_tokens > 0:
            encoding = self.tokenizer(
                current_chunk,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            chunks.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
        
        # Limit the number of chunks to avoid memory issues
        if len(chunks) > self.max_chunks:
            chunks = chunks[:self.max_chunks]
            
        return chunks
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Get text chunks
        chunks = self.chunk_text(text)
        
        return {
            'chunks': chunks,
            'label': torch.tensor(label, dtype=torch.long),
            'num_chunks': len(chunks)
        }

# Collate function for batching hierarchical data
def collate_hierarchical_batch(batch):
    all_chunks = []
    all_labels = []
    chunk_counts = []
    
    for item in batch:
        all_chunks.extend(item['chunks'])
        all_labels.append(item['label'])
        chunk_counts.append(item['num_chunks'])
    
    input_ids = torch.stack([chunk['input_ids'] for chunk in all_chunks])
    attention_mask = torch.stack([chunk['attention_mask'] for chunk in all_chunks])
    labels = torch.stack(all_labels)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'chunk_counts': chunk_counts
    }

# Save checkpoint function
def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, global_step, val_loss, output_dir, is_best=False):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_prefix = "best_model" if is_best else f"checkpoint_epoch{epoch}_step{global_step}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_prefix)
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(checkpoint_path, "model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_path)
    
    # Save optimizer and scheduler states
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'global_step': global_step,
        'val_loss': val_loss,
    }, os.path.join(checkpoint_path, "training_state.pt"))
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

# Load checkpoint function
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt")))
    
    # Load optimizer and scheduler states
    training_state = torch.load(os.path.join(checkpoint_path, "training_state.pt"))
    optimizer.load_state_dict(training_state['optimizer'])
    if scheduler and 'scheduler' in training_state and training_state['scheduler']:
        scheduler.load_state_dict(training_state['scheduler'])
    
    epoch = training_state['epoch']
    global_step = training_state['global_step']
    val_loss = training_state['val_loss']
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, global_step {global_step})")
    return epoch, global_step, val_loss

# Training function with hierarchical processing, cosine similarity loss, and manual stop
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, tokenizer, device, 
                epochs=3, gradient_accumulation_steps=4, eval_every=1000, 
                output_dir="models", resume_from=None):
    global STOP_TRAINING
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0
    
    # Resume from checkpoint if specified
    if resume_from:
        try:
            start_epoch, global_step, val_loss = load_checkpoint(
                model, optimizer, scheduler, resume_from
            )
            best_val_loss = val_loss
            logger.info(f"Resuming training from epoch {start_epoch}, step {global_step}, val_loss {val_loss:.4f}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        running_loss = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        optimizer.zero_grad()  # Zero gradients at the start of each epoch
        
        for step, batch in enumerate(progress_bar):
            # Check if manual stop signal received
            if STOP_TRAINING:
                logger.info("Manual stop requested. Saving current model...")
                current_model_path = save_checkpoint(
                    model, tokenizer, optimizer, scheduler, 
                    epoch, global_step, best_val_loss, output_dir, is_best=False
                )
                logger.info(f"Current model saved to {current_model_path}")
                logger.info("Stopping training...")
                return model
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            chunk_counts = batch['chunk_counts']
            
            # Forward pass to get embeddings
            embeddings = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Process hierarchical chunks by averaging embeddings from same document
            start_idx = 0
            doc_embeddings = []
            doc_labels = []
            
            for count, label in zip(chunk_counts, labels):
                if count > 0:
                    # Average embeddings from chunks of the same document
                    doc_embedding = embeddings[start_idx:start_idx + count].mean(dim=0, keepdim=True)
                    doc_embeddings.append(doc_embedding)
                    doc_labels.append(label)
                    start_idx += count
            
            if doc_embeddings:
                doc_embeddings = torch.cat(doc_embeddings, dim=0)
                doc_labels = torch.tensor(doc_labels, device=device)
                
                # Calculate loss using cosine similarity
                loss = criterion(doc_embeddings, doc_labels) / gradient_accumulation_steps
                loss.backward()
                
                train_loss += loss.item() * gradient_accumulation_steps
                running_loss += loss.item() * gradient_accumulation_steps
            
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
                    val_loss = evaluate_model(model, val_dataloader, criterion, device)
                    
                    logger.info(f"Step {global_step} - Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, tokenizer, optimizer, scheduler, 
                            epoch, global_step, val_loss, output_dir, is_best=True
                        )
                        logger.info(f"Step {global_step} - New best model saved with val loss: {best_val_loss:.4f}")
                    
                    ## Also save periodic checkpoint
                    #if global_step % (eval_every * 2) == 0:
                    #    save_checkpoint(
                    #        model, tokenizer, optimizer, scheduler, 
                    #        epoch, global_step, val_loss, output_dir, is_best=False
                    #    )

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
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} - Avg training loss: {avg_train_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Full validation at the end of each epoch
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(
            model, tokenizer, optimizer, scheduler, 
            epoch, global_step, val_loss, output_dir, is_best=False
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, tokenizer, optimizer, scheduler, 
                epoch, global_step, val_loss, output_dir, is_best=True
            )
            logger.info(f"Epoch {epoch + 1} - New best model saved with val loss: {best_val_loss:.4f}")
        
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load the best model at the end of training
    best_model_path = os.path.join(output_dir, "checkpoints", "best_model")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(os.path.join(best_model_path, "model.pt")))
        logger.info(f"Loaded best model from {best_model_path}")
    
    return model

# Evaluation function for embeddings
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            chunk_counts = batch['chunk_counts']
            
            # Get embeddings
            embeddings = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Process hierarchical chunks
            start_idx = 0
            doc_embeddings = []
            doc_labels = []
            
            for count, label in zip(chunk_counts, labels):
                if count > 0:
                    # Average embeddings for chunks from the same document
                    doc_embedding = embeddings[start_idx:start_idx + count].mean(dim=0, keepdim=True)
                    doc_embeddings.append(doc_embedding)
                    doc_labels.append(label)
                    start_idx += count
            
            if doc_embeddings:
                doc_embeddings = torch.cat(doc_embeddings, dim=0)
                doc_labels = torch.tensor(doc_labels, device=device)
                
                loss = criterion(doc_embeddings, doc_labels)
                val_loss += loss.item()
                
                all_embeddings.append(doc_embeddings.cpu())
                all_labels.extend(doc_labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(dataloader)
    
    # Additional metrics for semantic similarity would be computed here
    # This could include clustering metrics, retrieval metrics, etc.
    
    return avg_val_loss

# Function to extract embeddings for a text
def get_embedding(model, tokenizer, text, max_length=512, device='cpu'):
    model.eval()
    
    # For short texts
    if len(tokenizer.tokenize(text)) <= max_length:
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
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        return embedding.cpu().numpy()
    
    # For longer texts, use hierarchical chunking and averaging with sentence boundaries
    else:
        # Split text into sentences
        sentence_delimiters = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_tokens = len(tokenizer.tokenize(sentence))
            
            # Handle very long single sentences
            if sentence_tokens > max_length:
                # Process the current chunk if we have one
                if current_tokens > 0:
                    encoding = tokenizer(
                        current_chunk,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    chunks.append({
                        'input_ids': encoding['input_ids'].to(device),
                        'attention_mask': encoding['attention_mask'].to(device)
                    })
                
                # Split the long sentence into overlapping chunks
                sentence_tokens = tokenizer.tokenize(sentence)
                stride = max_length - 100  # 100 token overlap
                
                for i in range(0, len(sentence_tokens), stride):
                    chunk_tokens = sentence_tokens[i:i + max_length]
                    if len(chunk_tokens) < 100:  # Skip very small chunks
                        continue
                        
                    chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                    encoding = tokenizer(
                        chunk_text,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    chunks.append({
                        'input_ids': encoding['input_ids'].to(device),
                        'attention_mask': encoding['attention_mask'].to(device)
                    })
                
                # Reset for next sentence
                current_chunk = ""
                current_tokens = 0
            
            # If adding this sentence would exceed max_length, start a new chunk
            elif current_tokens + sentence_tokens > max_length:
                encoding = tokenizer(
                    current_chunk,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                chunks.append({
                    'input_ids': encoding['input_ids'].to(device),
                    'attention_mask': encoding['attention_mask'].to(device)
                })
                
                # Start new chunk with this sentence
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # Add sentence to current chunk
                if current_chunk:  # Add space if not the first sentence in chunk
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk and current_tokens > 0:
            encoding = tokenizer(
                current_chunk,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            chunks.append({
                'input_ids': encoding['input_ids'].to(device),
                'attention_mask': encoding['attention_mask'].to(device)
            })
        
        # Get embedding for each chunk
        chunk_embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                emb = model(
                    input_ids=chunk['input_ids'],
                    attention_mask=chunk['attention_mask']
                )
                chunk_embeddings.append(emb)
        
        # Average embeddings from all chunks
        if chunk_embeddings:
            avg_embedding = torch.mean(torch.cat(chunk_embeddings, dim=0), dim=0, keepdim=True)
            return avg_embedding.cpu().numpy()
        else:
            return None

# Main function to load data and train model
def main():
    # Configuration
    model_name = "emilyalsentzer/Bio_ClinicalBERT"  # Medical BERT model
    batch_size = 2  # Smaller batch size due to hierarchical processing
    epochs = 5
    learning_rate = 1e-5
    max_length = 512  # Max length for BERT
    embedding_dim = 768  # Output embedding dimension
    margin = 0.5  # Margin for cosine similarity loss (sufficiantly different if similarity of classes is below 0.5)
    gradient_accumulation_steps = 16  # Effective batch size = 2 × 16 = 32
    output_dir = "bertmodels/semantic_embeddings_with_sentencesplit"
    resume_from = None  # Set to checkpoint path to resume training
    
    logger.info("Press Ctrl+C at any time to stop training and save the current model")
    
    # Calculate training steps for learning rate scheduler
    total_samples = 179836  # Based on your dataset size
    train_size = int(0.8 * total_samples)  # 80% for training
    steps_per_epoch = math.ceil(train_size / (batch_size * gradient_accumulation_steps))
    total_training_steps = steps_per_epoch * epochs
    
    logger.info(f"Total training steps: {total_training_steps}")
    
    # Load data
    try:
        data = pd.read_csv("notes/train_data.csv")
        texts = data['text'].values 
        labels = data['Y'].values
        print(labels.shape)
        print(texts.shape)
        print(set(labels))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Example for demonstration - replace with your actual data
        logger.warning("Using dummy data for demonstration.")
        texts = ["Patient presents with symptoms..." for _ in range(100)]
        labels = np.random.randint(0, 2, size=100)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training on {len(train_texts)} examples, validating on {len(val_texts)} examples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create custom model for embeddings
    model = BertEmbeddingModel(model_name)
    
    # Create datasets with hierarchical chunking
    train_dataset = HierarchicalMedicalNotesDataset(
        train_texts, train_labels, tokenizer, max_length=max_length, overlap=100, max_chunks=5 
    )
    val_dataset = HierarchicalMedicalNotesDataset(
        val_texts, val_labels, tokenizer, max_length=max_length, overlap=100, max_chunks=5
    )
    
    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_hierarchical_batch
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_hierarchical_batch
    )
    
    # Create cosine similarity loss
    criterion = CosineSimilarityLoss(margin=margin)
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_training_steps),  # 6% warmup
        num_training_steps=total_training_steps
    )
    
    # Move model and criterion to device
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Train model with manual stopping capability
    model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_every=steps_per_epoch // 2,  # Evaluate 1 times per epoch
        output_dir=output_dir,
        resume_from=resume_from
    )
    
    # Final save path
    final_model_path = os.path.join(output_dir, "Trial_2_cos")
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path)
    
    # Save final model
    tokenizer.save_pretrained(final_model_path)
    torch.save(model.state_dict(), os.path.join(final_model_path, "model.pt"))
    
    logger.info(f"Final model saved to {final_model_path}")
    
    # Example function to calculate similarity between two texts
    def calculate_similarity(text1, text2):
        embedding1 = get_embedding(model, tokenizer, text1, device=device)
        embedding2 = get_embedding(model, tokenizer, text2, device=device)
        
        if embedding1 is not None and embedding2 is not None:
            # Calculate cosine similarity
            similarity = np.dot(embedding1.flatten(), embedding2.flatten()) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return similarity
        return None
    
    # Example usage
    example1 = "Patient shows signs of improvement after treatment with antibiotics."
    example2 = "After antibiotic therapy, patient's condition has improved significantly."
    example3 = "Patient diagnosed with type 2 diabetes and prescribed metformin."
    
    sim1_2 = calculate_similarity(example1, example2)
    sim1_3 = calculate_similarity(example1, example3)
    
    logger.info(f"Similarity between related examples: {sim1_2:.4f}")
    logger.info(f"Similarity between unrelated examples: {sim1_3:.4f}")

if __name__ == "__main__":
    main()