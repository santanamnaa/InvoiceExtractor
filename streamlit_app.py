import streamlit as st
import os
import tempfile
import json
from extraction_service import InvoiceExtractor
try:
    from enhanced_extraction_service import EnhancedInvoiceExtractor
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Initialize extractor
if ENHANCED_AVAILABLE:
    try:
        extractor = EnhancedInvoiceExtractor()
    except Exception:
        extractor = InvoiceExtractor()
else:
    extractor = InvoiceExtractor()

st.set_page_config(page_title="Invoice Extractor", layout="centered")
st.title("Invoice Extractor")
st.write("Upload an invoice file (PDF, image, or TXT) or paste OCR text to extract structured data.")

# File uploader
uploaded_file = st.file_uploader("Upload file (PDF, PNG, JPG, JPEG, TXT)", type=["pdf", "png", "jpg", "jpeg", "txt"])

# Text area for OCR text
ocr_text = st.text_area("Or paste OCR text here", height=200)

extract_btn = st.button("Extract Data")

if extract_btn:
    extracted_data = None
    error = None
    if uploaded_file is not None:
        # Save uploaded file to a temp file
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        try:
            extracted_data = extractor.extract_from_file(tmp_path)
        except Exception as e:
            error = f"Error extracting from file: {str(e)}"
        finally:
            os.remove(tmp_path)
    elif ocr_text.strip():
        try:
            extracted_data = extractor.extract_from_text(ocr_text)
        except Exception as e:
            error = f"Error extracting from text: {str(e)}"
    else:
        error = "Please upload a file or paste OCR text."

    if error:
        st.error(error)
    elif extracted_data:
        st.success("Extraction successful!")
        st.subheader("Extracted Data")
        st.json(extracted_data)
        st.download_button(
            label="Download JSON",
            data=json.dumps(extracted_data, indent=2, ensure_ascii=False),
            file_name="extracted_invoice.json",
            mime="application/json"
        )
    else:
        st.warning("No data could be extracted from the provided input.") 