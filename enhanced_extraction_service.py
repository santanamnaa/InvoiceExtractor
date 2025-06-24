"""
Enhanced Extraction Service with ML Integration and Telkom-Specific Patterns
Combines regex patterns with trained ML models for improved accuracy
Includes specialized extraction methods for Indonesian Telkom invoices
"""

import re
import logging
import pickle
import os
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from extraction_service import InvoiceExtractor

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. OCR from images will not work.")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pdfplumber not available. PDF processing will not work.")


class DataNormalizer:
    """Post-processing data normalization for extracted invoice data"""
    
    def __init__(self):
        self.date_patterns = [
            (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', 'DD/MM/YYYY'),
            (r'(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})', 'YYYY/MM/DD'),
            (r'(\d{1,2})\s+(\w+)\s+(\d{4})', 'DD Month YYYY')
        ]
        
        self.currency_symbols = ['Rp', 'IDR', 'Rupiah', 'rp', 'idr']
        self.npwp_pattern = r'\d{2}\.\d{3}\.\d{3}\.\d{1}\-\d{3}\.\d{3}'
    
    def normalize_date(self, date_str: str) -> str:
        """Normalize date to ISO format (YYYY-MM-DD)"""
        if not date_str or date_str.strip() == '':
            return ''
        
        for pattern, format_type in self.date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if format_type == 'DD/MM/YYYY':
                        day, month, year = match.groups()
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    elif format_type == 'YYYY/MM/DD':
                        year, month, day = match.groups()
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except (ValueError, IndexError):
                    continue
        
        return date_str
    
    def normalize_currency(self, amount_str: str) -> str:
        """Normalize currency amount (remove symbols, standardize format)"""
        if not amount_str or amount_str.strip() == '':
            return ''
        
        # Remove currency symbols
        cleaned = amount_str
        for symbol in self.currency_symbols:
            cleaned = cleaned.replace(symbol, '').strip()
        
        # Remove dots and commas, keep only digits
        cleaned = re.sub(r'[^\d]', '', cleaned)
        
        return cleaned if cleaned.isdigit() else amount_str
    
    def normalize_npwp(self, npwp_str: str) -> str:
        """Normalize NPWP to standard format"""
        if not npwp_str or npwp_str.strip() == '':
            return ''
        
        # Extract only digits
        digits = re.sub(r'[^\d]', '', npwp_str)
        
        if len(digits) == 15:
            # Format: XX.XXX.XXX.X-XXX.XXX
            return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}.{digits[8]}-{digits[9:12]}.{digits[12:15]}"
        
        return npwp_str
    
    def normalize_phone(self, phone_str: str) -> str:
        """Normalize phone numbers"""
        if not phone_str:
            return ''
        
        # Extract only digits and common separators
        cleaned = re.sub(r'[^\d\-\+\(\)\s]', '', phone_str)
        return cleaned.strip()


class EnhancedInvoiceExtractor(InvoiceExtractor):
    """Enhanced extractor combining regex patterns with ML models"""
    
    def __init__(self, model_path: str = 'models'):
        super().__init__()
        self.model_path = model_path
        self.ml_model = None
        self.feature_extractor = None
        self.normalizer = DataNormalizer()
        
        # Load ML model if available
        self._load_ml_models()
    
    def _load_ml_models(self):
        """Load pre-trained ML models"""
        try:
            # Load best model
            model_files = [f for f in os.listdir(self.model_path) if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                with open(os.path.join(self.model_path, model_files[0]), 'rb') as f:
                    self.ml_model = pickle.load(f)
                logging.info(f"Loaded ML model: {model_files[0]}")
            
            # Load feature extractor
            feature_extractor_path = os.path.join(self.model_path, 'feature_extractor.pkl')
            if os.path.exists(feature_extractor_path):
                with open(feature_extractor_path, 'rb') as f:
                    self.feature_extractor = pickle.load(f)
                logging.info("Loaded feature extractor")
            
        except Exception as e:
            logging.warning(f"Could not load ML models: {e}")
            self.ml_model = None
            self.feature_extractor = None
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Enhanced extraction using both regex and ML models"""
        if not text or not text.strip():
            raise Exception("No text provided for extraction")
        
        # Start with base regex extraction
        result = super().extract_from_text(text)
        
        # Enhance with ML if available
        if self.ml_model and self.feature_extractor:
            try:
                ml_results = self._extract_with_ml(text)
                result = self._merge_results(result, ml_results)
            except Exception as e:
                logging.warning(f"ML extraction failed: {e}")
        
        # Apply post-processing normalization
        result = self._apply_normalization(result)
        
        # Calculate confidence scores
        result['extraction_metadata'] = self._calculate_confidence(result, text)
        
        return result
    
    def _extract_with_ml(self, text: str) -> Dict[str, Any]:
        """Extract entities using trained ML model"""
        if not self.ml_model or not self.feature_extractor:
            return {}
        
        try:
            # Tokenize text
            from ml_trainer import IndonesianTokenizer
            tokenizer = IndonesianTokenizer()
            tokens = tokenizer.tokenize(text)
            
            # Extract features for each token using simple feature extraction
            features_list = []
            for i, token in enumerate(tokens):
                # Use simple feature extraction compatible with trained model
                features = self._extract_simple_features(tokens, i)
                features_list.append(features)
            
            if not features_list:
                return {}
            
            # Predict labels
            X = np.array(features_list)
            y_pred = self.ml_model.predict(X)
            
            # Convert predictions back to labels
            if hasattr(self.feature_extractor, 'label_encoder'):
                predicted_labels = self.feature_extractor['label_encoder'].inverse_transform(y_pred)
            elif isinstance(self.feature_extractor, dict) and 'label_encoder' in self.feature_extractor:
                predicted_labels = self.feature_extractor['label_encoder'].inverse_transform(y_pred)
            else:
                return {}
            
            # Extract entities from predictions
            ml_entities = self._extract_entities_from_predictions(tokens, predicted_labels)
            
            return ml_entities
            
        except Exception as e:
            logging.warning(f"ML extraction error: {e}")
            return {}
    
    def _extract_simple_features(self, tokens: List[str], position: int) -> List[float]:
        """Extract simple features compatible with trained model"""
        if position >= len(tokens):
            return [0.0] * 9
        
        token = tokens[position]
        features = [
            len(token),  # token length
            float(token.isdigit()),  # is numeric
            float(token.isalpha()),  # is alphabetic
            float('/' in token),  # has slash (dates)
            float('-' in token),  # has hyphen
            float('.' in token),  # has dot
            float(position < 3),  # is beginning
            float(position > len(tokens) - 3),  # is end
            position / len(tokens) if len(tokens) > 0 else 0,  # relative position
        ]
        return features
    
    def _extract_entities_from_predictions(self, tokens: List[str], labels: List[str]) -> Dict[str, Any]:
        """Convert ML predictions to structured entities"""
        entities = {
            'invoice_number': '',
            'invoice_date': '',
            'vendor_name': '',
            'buyer_name': '',
            'amount': '',
            'tax_id': '',
            'ml_confidence': 0.0
        }
        
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                # Save previous entity
                if current_entity and current_tokens:
                    entity_value = ' '.join(current_tokens)
                    if current_entity == 'INVOICE_NUMBER':
                        entities['invoice_number'] = entity_value
                    elif current_entity == 'INVOICE_DATE':
                        entities['invoice_date'] = entity_value
                    elif current_entity == 'VENDOR_NAME':
                        entities['vendor_name'] = entity_value
                    elif current_entity == 'BUYER_NAME':
                        entities['buyer_name'] = entity_value
                    elif current_entity == 'AMOUNT':
                        entities['amount'] = entity_value
                    elif current_entity == 'VENDOR_TAX_ID' or current_entity == 'BUYER_TAX_ID':
                        entities['tax_id'] = entity_value
                
                # Start new entity
                current_entity = label[2:]  # Remove 'B-' prefix
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
                
            else:
                # End current entity
                if current_entity and current_tokens:
                    entity_value = ' '.join(current_tokens)
                    if current_entity == 'INVOICE_NUMBER':
                        entities['invoice_number'] = entity_value
                    elif current_entity == 'INVOICE_DATE':
                        entities['invoice_date'] = entity_value
                    elif current_entity == 'VENDOR_NAME':
                        entities['vendor_name'] = entity_value
                    elif current_entity == 'BUYER_NAME':
                        entities['buyer_name'] = entity_value
                    elif current_entity == 'AMOUNT':
                        entities['amount'] = entity_value
                    elif current_entity in ['VENDOR_TAX_ID', 'BUYER_TAX_ID']:
                        entities['tax_id'] = entity_value
                
                current_entity = None
                current_tokens = []
        
        # Handle last entity
        if current_entity and current_tokens:
            entity_value = ' '.join(current_tokens)
            if current_entity == 'INVOICE_NUMBER':
                entities['invoice_number'] = entity_value
            elif current_entity == 'INVOICE_DATE':
                entities['invoice_date'] = entity_value
            elif current_entity == 'VENDOR_NAME':
                entities['vendor_name'] = entity_value
            elif current_entity == 'BUYER_NAME':
                entities['buyer_name'] = entity_value
            elif current_entity == 'AMOUNT':
                entities['amount'] = entity_value
            elif current_entity in ['VENDOR_TAX_ID', 'BUYER_TAX_ID']:
                entities['tax_id'] = entity_value
        
        return entities
    
    def _merge_results(self, regex_result: Dict, ml_result: Dict) -> Dict[str, Any]:
        """Merge regex and ML extraction results with confidence weighting"""
        merged = regex_result.copy()
        
        # Use ML results to fill in missing fields or improve existing ones
        if ml_result.get('invoice_number') and not merged.get('invoice_number'):
            merged['invoice_number'] = ml_result['invoice_number']
        
        if ml_result.get('invoice_date') and not merged.get('invoice_date'):
            merged['invoice_date'] = ml_result['invoice_date']
        
        if ml_result.get('vendor_name') and not merged['vendor']['name']:
            merged['vendor']['name'] = ml_result['vendor_name']
        
        if ml_result.get('buyer_name') and not merged['buyer']['name']:
            merged['buyer']['name'] = ml_result['buyer_name']
        
        if ml_result.get('amount') and not merged.get('invoice_total'):
            merged['invoice_total'] = ml_result['amount']
        
        if ml_result.get('tax_id'):
            if not merged['vendor']['tax_id']:
                merged['vendor']['tax_id'] = ml_result['tax_id']
            elif not merged['buyer']['tax_id']:
                merged['buyer']['tax_id'] = ml_result['tax_id']
        
        return merged
    
    def _apply_normalization(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply post-processing normalization to extracted data"""
        # Normalize dates
        if result.get('invoice_date'):
            result['invoice_date'] = self.normalizer.normalize_date(result['invoice_date'])
        
        if result.get('due_date'):
            result['due_date'] = self.normalizer.normalize_date(result['due_date'])
        
        if result.get('tax', {}).get('tax_issue_date'):
            result['tax']['tax_issue_date'] = self.normalizer.normalize_date(result['tax']['tax_issue_date'])
        
        # Normalize currency amounts
        if result.get('invoice_total'):
            result['invoice_total'] = self.normalizer.normalize_currency(result['invoice_total'])
        
        if result.get('tax', {}).get('subtotal'):
            result['tax']['subtotal'] = self.normalizer.normalize_currency(result['tax']['subtotal'])
        
        if result.get('tax', {}).get('tax_amount'):
            result['tax']['tax_amount'] = self.normalizer.normalize_currency(result['tax']['tax_amount'])
        
        # Normalize NPWP
        if result.get('vendor', {}).get('tax_id'):
            result['vendor']['tax_id'] = self.normalizer.normalize_npwp(result['vendor']['tax_id'])
        
        if result.get('buyer', {}).get('tax_id'):
            result['buyer']['tax_id'] = self.normalizer.normalize_npwp(result['buyer']['tax_id'])
        
        # Normalize line item amounts
        for item in result.get('line_items', []):
            if item.get('amount'):
                item['amount'] = self.normalizer.normalize_currency(item['amount'])
        
        return result
    
    def _calculate_confidence(self, result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Calculate extraction confidence scores"""
        metadata = {
            'extraction_method': 'hybrid_regex_ml' if self.ml_model else 'regex_only',
            'text_length': len(original_text),
            'extracted_fields': 0,
            'confidence_score': 0.0,
            'field_confidence': {},
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Count extracted fields
        key_fields = [
            'invoice_number', 'invoice_date', 'invoice_total',
            'vendor.name', 'vendor.tax_id', 'buyer.name'
        ]
        
        extracted_count = 0
        field_scores = {}
        
        for field in key_fields:
            if '.' in field:
                main_key, sub_key = field.split('.')
                value = result.get(main_key, {}).get(sub_key)
            else:
                value = result.get(field)
            
            if value and str(value).strip():
                extracted_count += 1
                # Calculate field confidence based on pattern matching
                confidence = self._calculate_field_confidence(field, value, original_text)
                field_scores[field] = confidence
        
        metadata['extracted_fields'] = extracted_count
        metadata['field_confidence'] = field_scores
        
        # Overall confidence score
        if extracted_count > 0:
            metadata['confidence_score'] = sum(field_scores.values()) / len(field_scores)
        
        return metadata
    
    def _calculate_field_confidence(self, field_name: str, value: str, text: str) -> float:
        """Calculate confidence score for individual field"""
        if not value or not value.strip():
            return 0.0
        
        # Pattern-based confidence scoring
        confidence = 0.5  # Base confidence
        
        if field_name == 'invoice_number':
            if re.match(r'[A-Z]{2,5}[-/]\d{4}[-/]\d{3,6}', value):
                confidence = 0.9
            elif re.match(r'[A-Z0-9\-/]{6,}', value):
                confidence = 0.7
        
        elif field_name == 'invoice_date':
            if re.match(r'\d{4}-\d{2}-\d{2}', value):  # ISO format
                confidence = 0.95
            elif re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', value):
                confidence = 0.8
        
        elif field_name == 'invoice_total':
            if value.isdigit() and len(value) >= 4:
                confidence = 0.8
            elif re.match(r'\d{3,}[,\.]\d{3}', value):
                confidence = 0.7
        
        elif 'tax_id' in field_name:
            if re.match(self.normalizer.npwp_pattern, value):
                confidence = 0.95
            elif re.match(r'\d{15}', value):
                confidence = 0.7
        
        elif 'name' in field_name:
            if any(prefix in value.upper() for prefix in ['PT', 'CV', 'UD']):
                confidence = 0.8
            elif len(value.split()) >= 2:
                confidence = 0.6
        
        # Boost confidence if value appears in original text
        if value.lower() in text.lower():
            confidence = min(1.0, confidence + 0.1)
        
        return round(confidence, 2)