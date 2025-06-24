# Invoice & Tax Document Extractor

## Overview

This is a Flask-based web application designed to extract structured information from Indonesian invoices and tax documents. The system uses a hybrid approach combining regex-based pattern matching with machine learning models for enhanced accuracy. It accepts PDF files, images (PNG, JPG, JPEG), or direct OCR text input and extracts key invoice data such as invoice numbers, dates, amounts, tax information, and vendor details.

## Enhanced ML Training Architecture

The system now includes comprehensive machine learning training capabilities:
- **Hybrid Regex+NER**: Combines regex patterns for fixed structures with ML-based named entity recognition
- **Layout-aware Features**: Leverages position and context information for better extraction accuracy
- **Indonesian Tokenization**: Custom tokenizer optimized for Indonesian invoice documents
- **Minimum 20 Training Iterations**: Ensures robust model training with early stopping
- **Cross-validation**: K-fold validation for stable performance estimation
- **Post-processing Normalization**: Automatic data standardization (dates to ISO format, currency normalization, NPWP formatting)

## System Architecture

### Frontend Architecture
- **Framework**: Bootstrap 5 with dark theme
- **Styling**: Custom CSS with modern card-based layout
- **JavaScript**: Vanilla JS for form validation and user interactions
- **Templates**: Jinja2 templating with Flask
- **Features**: 
  - File upload with drag-and-drop interface
  - OCR text input alternative
  - Real-time form validation
  - Responsive design optimized for various screen sizes

### Backend Architecture
- **Framework**: Flask web framework
- **Language**: Python 3.11
- **Pattern**: MVC architecture with service layer
- **Key Components**:
  - `app.py`: Main Flask application with routes
  - `extraction_service.py`: Core business logic for invoice data extraction
  - `main.py`: Application entry point

### Document Processing Pipeline
1. **Input Handling**: Accepts PDF files, images, or direct OCR text
2. **OCR Processing**: Uses Tesseract OCR for image-to-text conversion
3. **PDF Processing**: Uses pdfplumber for PDF text extraction
4. **Pattern Matching**: Regex-based extraction using Indonesian and English patterns
5. **Data Structuring**: Converts extracted text to structured JSON format

## Key Components

### Enhanced Extraction Services

#### InvoiceExtractor Service (Base)
- **Purpose**: Core regex-based extraction engine
- **Capabilities**: Basic pattern matching for invoice elements
- **Pattern Library**: Extensive regex patterns optimized for Indonesian invoice formats

#### EnhancedInvoiceExtractor Service (ML-Enhanced)
- **Purpose**: Hybrid extraction combining regex and machine learning
- **ML Models**: Trained Random Forest, Gradient Boosting, and Logistic Regression models
- **Features**: 
  - Layout-aware token features (position, context, patterns)
  - Indonesian-specific linguistic features
  - Confidence scoring for extracted data
  - Post-processing normalization
- **Training**: Minimum 20 iterations with early stopping and cross-validation

#### ML Training Pipeline
- **Data Preparation**: IOB2 format labeling, data augmentation, cross-validation splits
- **Feature Engineering**: 25+ layout-aware features per token including position, context, and pattern matching
- **Model Training**: Multiple algorithms with hyperparameter randomization
- **Evaluation**: Comprehensive error analysis, precision/recall metrics, false positive/negative analysis
- **Artifacts**: Trained models, feature extractors, training history, and evaluation reports

### File Upload System
- **Supported Formats**: PDF, PNG, JPG, JPEG, TXT
- **Size Limit**: 16MB maximum file size
- **Security**: Secure filename handling with werkzeug
- **Validation**: Client and server-side file type validation

### OCR Integration
- **Engine**: Tesseract OCR with pytesseract wrapper
- **Image Processing**: PIL/Pillow for image preprocessing
- **Fallback**: Graceful degradation when OCR dependencies unavailable

## Data Flow

1. **Upload Phase**: User uploads document or enters OCR text
2. **Processing Phase**: 
   - File type detection and validation
   - OCR processing for images
   - PDF text extraction for PDF files
   - Text preprocessing and normalization
3. **Extraction Phase**:
   - Pattern matching against predefined regex library
   - Data validation and cleaning
   - Structure formatting into JSON
4. **Display Phase**: Structured data presentation in web interface

## External Dependencies

### Core Dependencies
- **Flask**: Web framework and routing
- **pytesseract**: OCR text extraction from images
- **pdfplumber**: PDF text extraction and processing
- **Pillow**: Image processing and manipulation
- **Werkzeug**: WSGI utilities and security helpers

### System Dependencies
- **Tesseract**: OCR engine (system-level dependency)
- **PostgreSQL**: Database system (available but not currently used)
- **Various image libraries**: freetype, libjpeg, libwebp, etc.

### Frontend Dependencies
- **Bootstrap 5**: UI framework with dark theme
- **Font Awesome**: Icon library
- **Custom CSS/JS**: Application-specific styling and behavior

## Deployment Strategy

### Production Deployment
- **WSGI Server**: Gunicorn with autoscale deployment target
- **Configuration**: Environment-based configuration management
- **Port Binding**: 0.0.0.0:5000 with reuse-port option
- **Process Management**: Gunicorn handles worker processes

### Development Environment
- **Hot Reload**: Gunicorn with --reload option for development
- **Debug Mode**: Flask debug mode enabled in development
- **Port Forwarding**: Automatic port detection and forwarding

### Infrastructure
- **Nix Environment**: Reproducible development environment
- **System Packages**: All required system dependencies managed via Nix
- **Python Environment**: Python 3.11 with uv package management

## Changelog

- June 24, 2025. Initial setup with basic Flask application and regex-based extraction
- June 24, 2025. Enhanced with comprehensive ML training pipeline:
  - Added Jupyter notebooks for data preparation, model training, and evaluation
  - Implemented hybrid regex+ML extraction approach
  - Created Indonesian custom tokenizer with layout-aware features
  - Built training system with minimum 20 iterations and early stopping
  - Added post-processing normalization for dates, currency, and NPWP
  - Integrated cross-validation and comprehensive error analysis
  - Created EnhancedInvoiceExtractor with confidence scoring

## User Preferences

Preferred communication style: Simple, everyday language.