// Invoice Extractor JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const ocrTextInput = document.getElementById('ocrTextInput');
    const submitBtn = document.getElementById('submitBtn');
    
    // Form validation and submission handling
    if (form) {
        form.addEventListener('submit', function(e) {
            const hasFile = fileInput.files.length > 0;
            const hasText = ocrTextInput.value.trim().length > 0;
            
            if (!hasFile && !hasText) {
                e.preventDefault();
                showAlert('Please either upload a file or enter OCR text.', 'warning');
                return;
            }
            
            if (hasFile && hasText) {
                e.preventDefault();
                showAlert('Please provide either a file or OCR text, not both.', 'warning');
                return;
            }
            
            // Show loading state
            showLoadingState();
        });
    }
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Clear OCR text when file is selected
                if (ocrTextInput) {
                    ocrTextInput.value = '';
                }
                
                // Validate file size
                if (file.size > 16 * 1024 * 1024) { // 16MB
                    showAlert('File size must be less than 16MB.', 'error');
                    this.value = '';
                    return;
                }
                
                // Validate file type
                const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg', 'text/plain'];
                if (!allowedTypes.includes(file.type)) {
                    showAlert('Invalid file type. Please upload PDF, PNG, JPG, JPEG, or TXT files.', 'error');
                    this.value = '';
                    return;
                }
                
                updateFileInputLabel(file.name);
            }
        });
    }
    
    // OCR text input handler
    if (ocrTextInput) {
        ocrTextInput.addEventListener('input', function() {
            if (this.value.trim().length > 0 && fileInput) {
                // Clear file input when text is entered
                fileInput.value = '';
                updateFileInputLabel('Choose file...');
            }
        });
    }
    
    // Auto-dismiss alerts
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

function showAlert(message, type) {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertContainer, container.firstChild);
    
    // Auto dismiss after 5 seconds
    setTimeout(function() {
        const alert = new bootstrap.Alert(alertContainer);
        alert.close();
    }, 5000);
}

function showLoadingState() {
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('uploadForm');
    
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Processing...
        `;
    }
    
    if (form) {
        form.classList.add('form-submitting');
    }
    
    // Show loading overlay
    showLoadingOverlay();
}

function showLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'spinner-overlay';
    overlay.innerHTML = `
        <div class="text-center">
            <div class="spinner-border spinner-border-lg text-info" role="status"></div>
            <div class="mt-3 text-light">
                <h5>Processing Document...</h5>
                <p>This may take a few moments depending on the document size.</p>
            </div>
        </div>
    `;
    
    document.body.appendChild(overlay);
}

function updateFileInputLabel(filename) {
    const label = document.querySelector('label[for="fileInput"]');
    if (label && filename) {
        label.innerHTML = `<i class="fas fa-file-check"></i>`;
        label.title = filename;
    }
}

// Utility functions for result page
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function() {
            showAlert('Copied to clipboard!', 'success');
        }).catch(function() {
            fallbackCopy(text);
        });
    } else {
        fallbackCopy(text);
    }
}

function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showAlert('Copied to clipboard!', 'success');
    } catch (err) {
        showAlert('Failed to copy to clipboard.', 'error');
    }
    
    document.body.removeChild(textArea);
}

// Handle drag and drop for file upload
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const dropZone = fileInput ? fileInput.closest('.input-group') : null;
    
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            dropZone.classList.add('border-info');
        }
        
        function unhighlight(e) {
            dropZone.classList.remove('border-info');
        }
        
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                updateFileInputLabel(files[0].name);
                
                // Clear OCR text
                const ocrTextInput = document.getElementById('ocrTextInput');
                if (ocrTextInput) {
                    ocrTextInput.value = '';
                }
            }
        }
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('uploadForm');
        if (form) {
            form.submit();
        }
    }
    
    // Escape to clear form
    if (e.key === 'Escape') {
        const fileInput = document.getElementById('fileInput');
        const ocrTextInput = document.getElementById('ocrTextInput');
        
        if (fileInput) fileInput.value = '';
        if (ocrTextInput) ocrTextInput.value = '';
        
        updateFileInputLabel('Choose file...');
    }
});
