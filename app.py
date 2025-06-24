import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import json
from extraction_service import InvoiceExtractor
try:
    from enhanced_extraction_service import EnhancedInvoiceExtractor
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize extraction service (enhanced if available)
if ENHANCED_AVAILABLE:
    try:
        extractor = EnhancedInvoiceExtractor()
        logging.info("Using enhanced ML-powered extraction service")
    except Exception as e:
        logging.warning(f"Could not load enhanced extractor: {e}")
        extractor = InvoiceExtractor()
        logging.info("Using basic regex extraction service")
else:
    extractor = InvoiceExtractor()
    logging.info("Using basic regex extraction service")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files and 'ocr_text' not in request.form:
            flash('No file or OCR text provided', 'error')
            return redirect(url_for('index'))
        
        extracted_data = None
        
        # Handle file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload PDF, PNG, JPG, JPEG, or TXT files.', 'error')
                return redirect(url_for('index'))
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                extracted_data = extractor.extract_from_file(filepath)
            except Exception as e:
                logging.error(f"Error extracting from file: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('index'))
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        # Handle OCR text input
        elif 'ocr_text' in request.form and request.form['ocr_text'].strip():
            ocr_text = request.form['ocr_text'].strip()
            try:
                extracted_data = extractor.extract_from_text(ocr_text)
            except Exception as e:
                logging.error(f"Error extracting from text: {str(e)}")
                flash(f'Error processing OCR text: {str(e)}', 'error')
                return redirect(url_for('index'))
        
        if extracted_data:
            return render_template('result.html', 
                                 extracted_data=extracted_data,
                                 json_data=json.dumps(extracted_data, indent=2, ensure_ascii=False))
        else:
            flash('No data could be extracted from the provided input.', 'warning')
            return redirect(url_for('index'))
            
    except Exception as e:
        logging.error(f"Unexpected error in upload_file: {str(e)}")
        flash(f'An unexpected error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """API endpoint for programmatic access"""
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    extracted_data = extractor.extract_from_file(filepath)
                    return jsonify({
                        'success': True,
                        'data': extracted_data
                    })
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
        
        elif request.json and 'text' in request.json:
            text = request.json['text']
            try:
                extracted_data = extractor.extract_from_text(text)
                return jsonify({
                    'success': True,
                    'data': extracted_data
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
        
        return jsonify({
            'success': False,
            'error': 'No valid input provided'
        }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
