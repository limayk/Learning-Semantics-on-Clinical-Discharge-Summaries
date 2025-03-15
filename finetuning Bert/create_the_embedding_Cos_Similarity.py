import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging
import os
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

# Custom model that outputs embeddings - reused from original code
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

def chunk_text(text, tokenizer, max_length=512, overlap=100, max_chunks=6):
    # If text is shorter than max_length, just use it directly
    if len(tokenizer.tokenize(text)) <= max_length:
        encoding = tokenizer(
            text,
            max_length=max_length,
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
        sentence_tokens = len(tokenizer.tokenize(sentence))
        
        # If a single sentence is longer than max_length, we'll need to truncate it
        if sentence_tokens > max_length:
            # If we have content in the current chunk, finalize it first
            if current_tokens > 0:
                encoding = tokenizer(
                    current_chunk,
                    max_length=max_length,
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
            long_tokens = tokenizer.tokenize(sentence)
            stride = max_length - overlap
            
            for i in range(0, len(long_tokens), stride):
                chunk_tokens = long_tokens[i:i + max_length]
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
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0)
                })
            
            # Reset the current chunk
            current_chunk = ""
            current_tokens = 0
            
        # If adding this sentence would exceed max_length, finalize current chunk and start a new one
        elif current_tokens + sentence_tokens > max_length:
            # Finalize current chunk
            encoding = tokenizer(
                current_chunk,
                max_length=max_length,
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
        encoding = tokenizer(
            current_chunk,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        chunks.append({
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        })
    
    # Limit the number of chunks to avoid memory issues
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        
    return chunks


# Function to extract embeddings for a text chunk
def get_embedding(model, tokenizer, text, max_length=512, device='cpu', max_chunks=6):
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

        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

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
            print("Warning error!")
            return None

def main():
    # Configuration - reusing model settings from original code
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    max_length = 512
    output_dir = "created Embeddings"
    model_path = "bertmodels/semantic_embeddings_with_sentencesplit/checkpoints/best_model"  # Adjust if needed
    max_chunks = 6
    overlap = 100
    
    logger.info("Loading test data...")
    try:
        data = pd.read_csv("notes/test_data.csv")  # Adjust path if needed
        texts = data['text'].values
        labels = data['Y'].values if 'Y' in data.columns else ["unknown"] * len(texts)
        
        logger.info(f"Loaded {len(texts)} samples from test data")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = BertEmbeddingModel(model_name)
    try:
        # Try to load saved model weights
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), weights_only=True))
        logger.info(f"Loaded model weights from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load saved model weights: {e}")
        logger.warning("Using initial model weights")
    
    model = model.to(device)
    
    # Process and calculate embeddings
    logger.info("Processing texts and calculating embeddings...")
    
    all_embeddings = []
    all_chunks = []
    all_labels = []
    all_indices = []
    
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if idx % 10 == 0:
            logger.info(f"Processing sample {idx}/{len(texts)}")
            
        # Create chunks using original method
        
        # Calculate embeddings for each chunk
        embedding = get_embedding(model, tokenizer, text, max_length, device)
        
        # Store results
        all_embeddings.append(embedding)
        all_labels.append(label)
        all_indices.append(idx)
    
    # Convert embeddings to DataFrame
    logger.info("Creating output DataFrame...")
    
    # Create a DataFrame with chunk info and labels
    all_embeddings = np.vstack(all_embeddings)
    features_df = pd.DataFrame(all_embeddings)
    features_df['Y'] = all_labels
    features_df['index'] = all_indices
    
    # Save the features to a CSV file
    features_csv_path = os.path.join(output_dir, 'our_bert_cos_similarity.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"Features saved to {features_csv_path}")
    
    print(f"Feature dimension: {all_embeddings.shape[1]}")
    print(f"Number of examples: {len(all_labels)}")

if __name__ == "__main__":
    main()