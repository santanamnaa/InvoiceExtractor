#!/usr/bin/env python3
"""
Enhanced ML Training System for Invoice Extraction
Integrates hybrid regex+NER approach with layout-aware features
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from extraction_service import InvoiceExtractor


class IndonesianTokenizer:
    """Custom tokenizer for Indonesian invoice documents with layout awareness"""
    
    def __init__(self):
        # Indonesian-specific patterns for invoice documents
        self.currency_pattern = r'(?:Rp\.?|IDR|Rupiah)\s*[\d,\.]+'  
        self.date_pattern = r'\d{1,2}[\-\/]\d{1,2}[\-\/]\d{2,4}'
        self.npwp_pattern = r'\d{2}\.\d{3}\.\d{3}\.\d{1}\-\d{3}\.\d{3}'
        self.company_pattern = r'(?:PT|CV|UD)\s+[A-Z\s]+'
        
        # Common Indonesian invoice terms
        self.indonesian_keywords = {
            'invoice', 'faktur', 'tagihan', 'pembayaran', 'total', 'jumlah', 
            'npwp', 'pajak', 'ppn', 'alamat', 'tanggal', 'nomor', 'bank'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization preserving Indonesian patterns"""
        if not text:
            return []
            
        # Preserve special patterns first
        special_tokens = []
        patterns = [
            (self.currency_pattern, 'CURRENCY_TOKEN'),
            (self.date_pattern, 'DATE_TOKEN'),
            (self.npwp_pattern, 'NPWP_TOKEN'),
            (self.company_pattern, 'COMPANY_TOKEN')
        ]
        
        processed_text = text
        for pattern, token_type in patterns:
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            for match in matches:
                special_token = f'<{token_type}_{len(special_tokens)}>'
                special_tokens.append((special_token, match.group()))
                processed_text = processed_text.replace(match.group(), special_token, 1)
        
        # Basic tokenization with word boundaries
        tokens = re.findall(r'\b\w+\b|[^\w\s]', processed_text)
        
        # Restore special tokens
        for special_token, original_text in special_tokens:
            for i, token in enumerate(tokens):
                if special_token in token:
                    tokens[i] = original_text
        
        return tokens


class LayoutAwareFeatureExtractor:
    """Extract layout-aware features for invoice NER training"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def extract_token_features(self, tokens: List[str], position_idx: int) -> Dict[str, float]:
        """Extract comprehensive features for a single token"""
        if position_idx >= len(tokens):
            return {}
            
        token = tokens[position_idx]
        features = {}
        
        # Basic token features
        features['token_length'] = len(token)
        features['is_numeric'] = float(token.isdigit())
        features['is_alpha'] = float(token.isalpha())
        features['is_alphanumeric'] = float(token.isalnum())
        features['is_upper'] = float(token.isupper())
        features['is_lower'] = float(token.islower())
        features['is_title'] = float(token.istitle())
        features['has_punctuation'] = float(any(c in '.,:-/\\' for c in token))
        
        # Layout-aware position features
        features['relative_position'] = position_idx / len(tokens) if len(tokens) > 0 else 0
        features['is_document_start'] = float(position_idx < 3)
        features['is_document_end'] = float(position_idx > len(tokens) - 3)
        features['is_document_middle'] = float(0.3 <= position_idx / len(tokens) <= 0.7)
        features['absolute_position'] = position_idx
        
        # Pattern-based features for Indonesian invoices
        features['is_currency_pattern'] = float(re.match(r'.*[Rr]p.*|.*IDR.*|.*\d+[,\.]?\d*.*', token) is not None)
        features['is_date_pattern'] = float(re.match(r'\d{1,2}[\-\/]\d{1,2}[\-\/]\d{2,4}', token) is not None)
        features['is_npwp_pattern'] = float(re.match(r'\d{2}\.\d{3}\.\d{3}\.\d{1}\-\d{3}\.\d{3}', token) is not None)
        features['is_company_prefix'] = float(token.upper() in ['PT', 'CV', 'UD'])
        features['is_invoice_keyword'] = float(token.lower() in ['invoice', 'faktur', 'tagihan', 'total', 'jumlah'])
        
        # Context features (neighboring tokens)
        if position_idx > 0:
            prev_token = tokens[position_idx - 1]
            features['prev_is_numeric'] = float(prev_token.isdigit())
            features['prev_is_colon'] = float(prev_token == ':')
            features['prev_is_currency'] = float(prev_token.lower() in ['rp', 'idr', 'rupiah'])
        else:
            features['prev_is_numeric'] = 0.0
            features['prev_is_colon'] = 0.0
            features['prev_is_currency'] = 0.0
            
        if position_idx < len(tokens) - 1:
            next_token = tokens[position_idx + 1]
            features['next_is_numeric'] = float(next_token.isdigit())
            features['next_is_colon'] = float(next_token == ':')
        else:
            features['next_is_numeric'] = 0.0
            features['next_is_colon'] = 0.0
        
        # Character-level features
        features['starts_with_digit'] = float(token[0].isdigit() if token else False)
        features['ends_with_digit'] = float(token[-1].isdigit() if token else False)
        features['has_hyphen'] = float('-' in token)
        features['has_slash'] = float('/' in token)
        features['has_dot'] = float('.' in token)
        
        return features
    
    def prepare_training_data(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels for training"""
        tokenizer = IndonesianTokenizer()
        features_list = []
        labels_list = []
        
        for sample in samples:
            tokens = tokenizer.tokenize(sample['text'])
            labels = sample.get('labels', ['O'] * len(tokens))
            
            # Ensure labels match token count
            while len(labels) < len(tokens):
                labels.append('O')
            labels = labels[:len(tokens)]
            
            for i, (token, label) in enumerate(zip(tokens, labels)):
                features = self.extract_token_features(tokens, i)
                features_list.append(list(features.values()))
                labels_list.append(label)
                
                # Store feature names for reference
                if not self.feature_names:
                    self.feature_names = list(features.keys())
        
        X = np.array(features_list)
        y = self.label_encoder.fit_transform(labels_list)
        
        return X, y, self.label_encoder.classes_.tolist()


class EnhancedInvoiceTrainer:
    """Enhanced training system with minimum 20 iterations and early stopping"""
    
    def __init__(self, min_iterations: int = 20):
        self.min_iterations = min_iterations
        self.training_history = []
        self.best_model = None
        self.best_f1_score = 0.0
        self.feature_extractor = LayoutAwareFeatureExtractor()
        
        # Model configurations for different iterations
        self.model_configs = [
            ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
            ('Logistic Regression', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ('Random Forest Deep', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)),
            ('Gradient Boosting Tuned', GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, random_state=42))
        ]
    
    def generate_sample_data(self, num_samples: int = 200) -> List[Dict]:
        """Generate comprehensive sample training data with IOB2 labels"""
        samples = []
        
        # Base templates for Indonesian invoices
        templates = [
            {
                'text': 'INVOICE NO: {inv_no} TANGGAL: {date} PT {company} NPWP: {npwp} TOTAL: Rp {amount}',
                'entities': [
                    {'start': 12, 'end': 12 + len('{inv_no}'), 'label': 'INVOICE_NUMBER'},
                    {'start': 21 + len('{inv_no}'), 'end': 21 + len('{inv_no}') + len('{date}'), 'label': 'INVOICE_DATE'},
                    {'start': 24 + len('{inv_no}') + len('{date}'), 'end': 24 + len('{inv_no}') + len('{date}') + len('{company}'), 'label': 'VENDOR_NAME'},
                    {'start': 30 + len('{inv_no}') + len('{date}') + len('{company}'), 'end': 30 + len('{inv_no}') + len('{date}') + len('{company}') + len('{npwp}'), 'label': 'VENDOR_TAX_ID'},
                    {'start': 38 + len('{inv_no}') + len('{date}') + len('{company}') + len('{npwp}'), 'end': 38 + len('{inv_no}') + len('{date}') + len('{company}') + len('{npwp}') + len('{amount}'), 'label': 'AMOUNT'}
                ]
            },
            {
                'text': 'FAKTUR PAJAK: {faktur_no} PERIODE: {period} CV {buyer} ALAMAT: {address}',
                'entities': [
                    {'start': 14, 'end': 14 + len('{faktur_no}'), 'label': 'FAKTUR_NUMBER'},
                    {'start': 23 + len('{faktur_no}'), 'end': 23 + len('{faktur_no}') + len('{period}'), 'label': 'BILLING_MONTH'},
                    {'start': 26 + len('{faktur_no}') + len('{period}'), 'end': 26 + len('{faktur_no}') + len('{period}') + len('{buyer}'), 'label': 'BUYER_NAME'},
                    {'start': 34 + len('{faktur_no}') + len('{period}') + len('{buyer}'), 'end': 34 + len('{faktur_no}') + len('{period}') + len('{buyer}') + len('{address}'), 'label': 'BUYER_ADDRESS'}
                ]
            }
        ]
        
        # Generate variations
        for i in range(num_samples):
            template = templates[i % len(templates)]
            
            # Generate realistic data
            inv_no = f"INV-2024-{1000 + i:04d}"
            date = f"{(i % 28) + 1:02d}/{(i % 12) + 1:02d}/2024"
            company = f"TELKOM INDONESIA" if i % 2 == 0 else "INDOSAT OOREDOO"
            npwp = f"{i%100:02d}.{(i*3)%999:03d}.{(i*7)%999:03d}.{i%9}-{(i*11)%999:03d}.{(i*13)%999:03d}"
            amount = f"{(i + 1) * 100000:,}"
            faktur_no = f"010.000-24.{i+1:08d}"
            period = f"2024{(i%12)+1:02d}"
            buyer = f"MAJU JAYA" if i % 2 == 0 else "SUKSES MANDIRI"
            address = f"JL SUDIRMAN NO {i+1} JAKARTA"
            
            # Fill template
            text = template['text'].format(
                inv_no=inv_no, date=date, company=company, npwp=npwp, amount=amount,
                faktur_no=faktur_no, period=period, buyer=buyer, address=address
            )
            
            # Convert to IOB2 format
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Simple IOB2 labeling (simplified for demo)
            for token_idx, token in enumerate(tokens):
                if 'INV-' in token:
                    labels[token_idx] = 'B-INVOICE_NUMBER'
                elif re.match(r'\d{2}/\d{2}/\d{4}', token):
                    labels[token_idx] = 'B-INVOICE_DATE'
                elif token in ['TELKOM', 'INDOSAT']:
                    labels[token_idx] = 'B-VENDOR_NAME'
                elif 'INDONESIA' in token or 'OOREDOO' in token:
                    labels[token_idx] = 'I-VENDOR_NAME'
                elif re.match(r'\d{2}\.\d{3}', token):
                    labels[token_idx] = 'B-VENDOR_TAX_ID'
                elif re.match(r'\d{3}\.\d{1}', token):
                    labels[token_idx] = 'I-VENDOR_TAX_ID'
                elif token.replace(',', '').isdigit() and len(token.replace(',', '')) > 4:
                    labels[token_idx] = 'B-AMOUNT'
                elif 'JAYA' in token or 'MANDIRI' in token:
                    labels[token_idx] = 'B-BUYER_NAME'
                elif token.startswith('JL'):
                    labels[token_idx] = 'B-BUYER_ADDRESS'
                elif 'JAKARTA' in token:
                    labels[token_idx] = 'I-BUYER_ADDRESS'
            
            sample = {
                'id': f'sample_{i:04d}',
                'text': text,
                'tokens': tokens,
                'labels': labels
            }
            samples.append(sample)
        
        return samples
    
    def train_with_iterations(self, samples: List[Dict]) -> Dict:
        """Run training with minimum 20 iterations and early stopping"""
        print(f"Starting enhanced training with minimum {self.min_iterations} iterations")
        print(f"Dataset size: {len(samples)} samples")
        
        # Prepare data
        X, y, label_classes = self.feature_extractor.prepare_training_data(samples)
        print(f"Feature matrix: {X.shape}, Labels: {len(np.unique(y))} classes")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        iteration = 0
        patience_counter = 0
        patience = 5
        
        # Run training iterations
        for round_num in range(5):  # Multiple rounds to ensure 20+ iterations
            for model_name, base_model in self.model_configs:
                iteration += 1
                
                print(f"\nIteration {iteration}: Training {model_name}")
                
                # Add randomization after first round
                if iteration > len(self.model_configs):
                    model = self._randomize_model(base_model)
                else:
                    model = base_model
                
                # Train model
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                train_f1 = f1_score(y_train, y_pred_train, average='weighted')
                val_f1 = f1_score(y_val, y_pred_val, average='weighted')
                
                # Per-class metrics
                precision, recall, f1_per_class, support = precision_recall_fscore_support(
                    y_val, y_pred_val, average=None, zero_division=0
                )
                
                # Store results
                result = {
                    'iteration': iteration,
                    'model': model_name,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'precision_per_class': precision.tolist(),
                    'recall_per_class': recall.tolist(),
                    'f1_per_class': f1_per_class.tolist(),
                    'training_time': training_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.training_history.append(result)
                
                # Update best model
                if val_f1 > self.best_f1_score:
                    self.best_f1_score = val_f1
                    self.best_model = (model_name, model)
                    patience_counter = 0
                    print(f"   New best model! F1: {val_f1:.4f}")
                else:
                    patience_counter += 1
                
                print(f"   Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
                print(f"   Training time: {training_time:.2f}s")
                
                # Early stopping conditions
                if iteration >= self.min_iterations:
                    if val_f1 > 0.9:
                        print(f"\nTarget F1 score (>0.9) achieved at iteration {iteration}!")
                        break
                    elif patience_counter >= patience:
                        print(f"\nEarly stopping at iteration {iteration} (patience exceeded)")
                        break
            
            if iteration >= self.min_iterations and (val_f1 > 0.9 or patience_counter >= patience):
                break
        
        # Training summary
        summary = {
            'total_iterations': iteration,
            'best_model_name': self.best_model[0] if self.best_model else None,
            'best_f1_score': self.best_f1_score,
            'target_achieved': self.best_f1_score > 0.9,
            'training_completed_at': datetime.now().isoformat(),
            'dataset_size': len(samples),
            'feature_count': X.shape[1],
            'label_classes': label_classes
        }
        
        print(f"\nTraining completed:")
        print(f"- Total iterations: {summary['total_iterations']}")
        print(f"- Best model: {summary['best_model_name']}")
        print(f"- Best F1 score: {summary['best_f1_score']:.4f}")
        print(f"- Target achieved (F1 > 0.9): {summary['target_achieved']}")
        
        return summary
    
    def _randomize_model(self, base_model):
        """Add randomization to model parameters"""
        model_type = type(base_model).__name__
        
        if model_type == 'RandomForestClassifier':
            return RandomForestClassifier(
                n_estimators=np.random.choice([50, 100, 200, 300]),
                max_depth=np.random.choice([10, 20, 30, None]),
                min_samples_split=np.random.choice([2, 5, 10]),
                random_state=np.random.randint(1, 100)
            )
        elif model_type == 'GradientBoostingClassifier':
            return GradientBoostingClassifier(
                n_estimators=np.random.choice([50, 100, 150, 200]),
                learning_rate=np.random.choice([0.05, 0.1, 0.2, 0.3]),
                max_depth=np.random.choice([3, 5, 7]),
                random_state=np.random.randint(1, 100)
            )
        elif model_type == 'LogisticRegression':
            return LogisticRegression(
                C=np.random.choice([0.1, 1.0, 10.0, 100.0]),
                max_iter=1000,
                random_state=np.random.randint(1, 100)
            )
        
        return base_model
    
    def save_training_artifacts(self, output_dir: str = 'models'):
        """Save all training artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.best_model:
            # Save best model
            model_name, model = self.best_model
            model_filename = f"best_model_{model_name.lower().replace(' ', '_')}.pkl"
            with open(os.path.join(output_dir, model_filename), 'wb') as f:
                pickle.dump(model, f)
            
            # Save feature extractor
            with open(os.path.join(output_dir, 'feature_extractor.pkl'), 'wb') as f:
                pickle.dump(self.feature_extractor, f)
            
            print(f"Saved best model: {model_filename}")
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save training summary
        summary = {
            'total_iterations': len(self.training_history),
            'best_model': self.best_model[0] if self.best_model else None,
            'best_f1_score': self.best_f1_score,
            'target_achieved': self.best_f1_score > 0.9,
            'feature_names': self.feature_extractor.feature_names,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training artifacts saved to {output_dir}/")


def main():
    """Main training execution"""
    print("Enhanced Invoice Extraction ML Training System")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EnhancedInvoiceTrainer(min_iterations=20)
    
    # Generate training data
    print("Generating sample training data...")
    samples = trainer.generate_sample_data(num_samples=300)
    print(f"Generated {len(samples)} training samples")
    
    # Run training
    summary = trainer.train_with_iterations(samples)
    
    # Save artifacts
    trainer.save_training_artifacts()
    
    print("\nTraining pipeline completed successfully!")
    return summary


if __name__ == "__main__":
    main()