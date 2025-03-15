import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
from tqdm import tqdm

# Function to run model on test data and provide detailed error analysis
def evaluate_on_test_data(model_path, test_data_path, output_file=None, batch_size=16, max_length=512):
    """
    Run the fine-tuned model on test data and provide detailed error analysis.
    
    Args:
        model_path (str): Path to the saved model directory
        test_data_path (str): Path to test data CSV
        output_file (str, optional): Path to save results
        batch_size (int): Batch size for inference
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Dictionary containing metrics and predictions
    """
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"usising devise: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load test data
    test_data = pd.read_csv(test_data_path) #creates the dataframe
    
    # Create test dataset
    class TestDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    # Prepare test dataset and loader
    test_texts = test_data['text'].values
    test_labels = test_data['Y'].values
    
    test_dataset = TestDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Run inference
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            batch_indices = list(range(i*batch_size, min((i+1)*batch_size, len(test_dataset))))
            all_indices.extend(batch_indices)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'index': all_indices,
        'text': [test_texts[i] for i in all_indices],
        'true_label': [test_labels[i] for i in all_indices],
        'predicted': all_preds,
        'prob_class_0': [p[0] for p in all_probs],
        'prob_class_1': [p[1] for p in all_probs]
    })
    
    # Add error analysis
    results_df['correct'] = (results_df['true_label'] == results_df['predicted']).astype(int)
    results_df['error_type'] = 'Correct'
    
    # Mark false positives and false negatives
    false_pos_mask = (results_df['true_label'] == 0) & (results_df['predicted'] == 1)
    false_neg_mask = (results_df['true_label'] == 1) & (results_df['predicted'] == 0)
    
    results_df.loc[false_pos_mask, 'error_type'] = 'False Positive'
    results_df.loc[false_neg_mask, 'error_type'] = 'False Negative'
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Print summary 
    print(f"Test Accuracy: {report['accuracy']:.4f}")
    print(f"Test F1-Score: {report['1']['f1-score']:.4f}")
    print(f"False Positive Rate: {cm[0, 1] / (cm[0, 0] + cm[0, 1]):.4f}")
    print(f"False Negative Rate: {cm[1, 0] / (cm[1, 0] + cm[1, 1]):.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # Sort errors by confidence for analysis
    fp_errors = results_df[false_pos_mask].sort_values(by='prob_class_1', ascending=False)
    fn_errors = results_df[false_neg_mask].sort_values(by='prob_class_0', ascending=False)
    
    # Save results if output file provided
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Save high confidence errors to separate files
        fp_file = output_file.replace('.csv', '_false_positives.csv')
        fn_file = output_file.replace('.csv', '_false_negatives.csv')
        
        fp_errors.to_csv(fp_file, index=False)
        fn_errors.to_csv(fn_file, index=False)
        print(f"False positive errors saved to {fp_file}")
        print(f"False negative errors saved to {fn_file}")
    
    return {
        'metrics': report,
        'confusion_matrix': cm,
        'results_df': results_df,
        'false_positives': fp_errors,
        'false_negatives': fn_errors
    }

# Function to extract features from the last layer and train a custom model
def extract_features_and_train_custom_model(model_path, data_path, output_dir, batch_size=16, max_length=512):
    """
    Extract embeddings from the last layer of the model and create a CSV file with 
    the token embeddings and labels for training a custom model.
    
    Args:
        model_path (str): Path to the saved model directory
        data_path (str): Path to data CSV file
        output_dir (str): Directory to save embeddings and custom model
        batch_size (int): Batch size for feature extraction
        max_length (int): Maximum sequence length
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model without classification head
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load data
    data = pd.read_csv(data_path)
    texts = data['text'].values
    labels = data['Y'].values
    print(texts[0])
    print(labels)
    print("textshape", texts.shape)
    print("labels shape", labels.shape)
    # Create dataset for feature extraction
    class FeatureExtractionDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    # Create dataset and dataloader
    dataset = FeatureExtractionDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Create a new model by removing the classification head
    class BertFeatureExtractor(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.bert = model.bert
            
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Last hidden state is what we want
            return outputs.last_hidden_state
    
    # Instantiate the feature extractor
    feature_extractor = BertFeatureExtractor(base_model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Extract features
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting Features")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Get the last hidden state
            features = feature_extractor(input_ids, attention_mask)
            
            # Get CLS token representation ([CLS] is the first token)
            cls_features = features[:, 0, :].cpu().numpy()
            
            all_features.append(cls_features)
            all_labels.extend(labels)
            
            batch_indices = list(range(i*batch_size, min((i+1)*batch_size, len(dataset))))
            all_indices.extend(batch_indices)
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    
    # Create a DataFrame with features and labels
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    features_df['index'] = all_indices
    #features_df['text'] = [texts[i] for i in all_indices]
    
    # Save the features to a CSV file
    features_csv_path = os.path.join(output_dir, 'our_bert_features.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"Features saved to {features_csv_path}")
    
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Number of examples: {len(all_labels)}")
    
    return features_csv_path

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual paths
    model_path = "bertmodels/Trial_one"
    test_data_path = "notes/test_data.csv"
    output_dir = "datasets" #???
    
    # Evaluate on test data
    #print("Evaluating model on test data...")
    #results = evaluate_on_test_data(
    #    model_path=model_path,
    #    test_data_path=test_data_path,
    #    output_file="test_results.csv",
    #    batch_size=16
    #)
    
    # Extract features and prepare for custom model
    print("\nExtracting features for custom model training...")
    features_csv = extract_features_and_train_custom_model(
        model_path=model_path,
        data_path=test_data_path,
        output_dir=output_dir,
        batch_size=16
    )
    
    print("\nFeature extraction complete. You can now train a custom linear model using:")
    print("features_df = pd.read_csv('" + features_csv + "')")
    print("X = features_df.drop(['label', 'index', 'text'], axis=1)")
    print("y = features_df['label']")
    print("model = LogisticRegression()")
    print("model.fit(X, y)")