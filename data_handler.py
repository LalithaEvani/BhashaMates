"""
Handle data loading and processing for intent classification
"""
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset


class IntentDataset(Dataset):
    """Dataset for intent classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class DataHandler:
    """Handle data loading, processing, and batching"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        self.dataset = None
        self.intent_names = None
        self.label_mapping = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    def get_max_tokenized_length(self,texts, tokenizer):
        lengths = [
            len(tokenizer(text, truncation=False, padding=False)["input_ids"])
            for text in texts
        ]
        return max(lengths)
    def load_data(self):
        """Load dataset"""
        if self.config.data.data_dir:
            # Load from local files
            train_df = pd.read_csv(os.path.join(self.config.data.data_dir, "train.csv"))
            test_df = pd.read_csv(os.path.join(self.config.data.data_dir, "test.csv"))
            
            # Extract texts and labels
            train_texts = train_df["text"].tolist()
            train_labels = train_df["label"].tolist()
            test_texts = test_df["text"].tolist()
            test_labels = test_df["label"].tolist()
            
            # Create datasets
            self.dataset = {
                "train": {"text": train_texts, "label": train_labels},
                "test": {"text": test_texts, "label": test_labels}
            }
            
            # Get unique intents and create mapping
            unique_intents = sorted(set(train_labels + test_labels))
            self.intent_names = unique_intents
            self.label_mapping = {intent: i for i, intent in enumerate(unique_intents)}
            
            # Map string labels to indices
            self.dataset["train"]["label"] = [self.label_mapping[label] for label in self.dataset["train"]["label"]]
            self.dataset["test"]["label"] = [self.label_mapping[label] for label in self.dataset["test"]["label"]]
        else:
            # Load from HuggingFace datasets
            dataset = load_dataset(self.config.data.dataset_name)
            self.dataset = dataset
            # print(dataset.keys()) train and test
            print(f'dataset {dataset}')
            # Get unique intents and create mapping
            if isinstance(dataset["train"], HFDataset):
                unique_intents = sorted(set(dataset["train"]["label"]))
                self.intent_names = unique_intents
                print(f'intent names {self.intent_names}')
                self.label_mapping = {intent: i for i, intent in enumerate(unique_intents)}
                
                # For Banking77, map 'intent' to 'label'
                if self.config.data.dataset_name == "banking77":
                    self.dataset = {
                        "train": {
                            "text": dataset["train"]["text"],
                            "label": [self.label_mapping[intent] for intent in dataset["train"]["label"]]
                        },
                        "test": {
                            "text": dataset["test"]["text"],
                            "label": [self.label_mapping[intent] for intent in dataset["test"]["label"]]
                        }
                    }
                        # Reverse label_mapping to get label index -> intent name
            inv_label_mapping = {v: k for k, v in self.label_mapping.items()}

            print("\nSample training examples:")
            sampled = []
            i = 0
            while(True):
                text = self.dataset["train"]["text"][i]
                label_idx = self.dataset["train"]["label"][i]
                i +=1
                if label_idx not in sampled:
                    if len(sampled) > 10:
                        break
                    sampled.append(label_idx)
                    label_str = inv_label_mapping[label_idx]
                    print(f"{i+1}. Text: {text}")
                    print(f"   Label index: {label_idx}  |  Label name: {label_str}\n")
                    max_len = self.get_max_tokenized_length(dataset["train"]["text"], self.tokenizer)
                    print(f"Maximum tokenized sequence length in the train set: {max_len}")
                    max_len = self.get_max_tokenized_length(dataset["test"]["text"], self.tokenizer)
                    print(f"Maximum tokenized sequence length in the test set: {max_len}")
    
    def prepare_data(self):
        """Prepare train/val/test datasets"""
        # Split train into train and val
        if isinstance(self.dataset["train"], dict):
            train_size = len(self.dataset["train"]["text"])
            val_size = int(0.1 * train_size)
            print(f'val size {val_size }')
            train_size = train_size - val_size
            print(f'train size {train_size}')
            
            # Split data
            indices = list(range(train_size + val_size))
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            # Create train and val datasets
            train_texts = [self.dataset["train"]["text"][i] for i in train_indices]
            train_labels = [self.dataset["train"]["label"][i] for i in train_indices]
            val_texts = [self.dataset["train"]["text"][i] for i in val_indices]
            val_labels = [self.dataset["train"]["label"][i] for i in val_indices]
            
            # Create test dataset
            test_texts = self.dataset["test"]["text"]
            test_labels = self.dataset["test"]["label"]
            print(f'test size {len(test_labels)}')

        else:
            # Handle HuggingFace Dataset objects
            train_val = self.dataset["train"].train_test_split(test_size=0.1)
            train_texts = train_val["train"]["text"]
            print(f'train size {len(train_texts)}')
            train_labels = [self.label_mapping[intent] for intent in train_val["train"]["label"]]
            val_texts = train_val["test"]["text"]
            val_labels = [self.label_mapping[intent] for intent in train_val["test"]["label"]]
            print(f'val size {len(val_labels)}')
            test_texts = self.dataset["test"]["text"]
            test_labels = [self.label_mapping[intent] for intent in self.dataset["test"]["label"]]
            print(f'test size {len(test_labels)}')
        
        # Create PyTorch datasets
        self.train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer, self.config.data.max_length)
        self.val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer, self.config.data.max_length)
        self.test_dataset = IntentDataset(test_texts, test_labels, self.tokenizer, self.config.data.max_length)
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_dataloaders(self):
        """Create DataLoaders for train, val, and test sets"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_num_labels(self):
        """Get number of intent labels"""
        return len(self.intent_names)
    
    def get_intent_names(self):
        """Get list of intent names"""
        return self.intent_names