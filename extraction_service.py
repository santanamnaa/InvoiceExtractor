import re
import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

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

class InvoiceExtractor:
    def __init__(self):
        self.patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for extracting invoice data"""
        return {
            # Invoice basic info
            'invoice_number': re.compile(r'(?:invoice|faktur|no\.?\s*invoice|nomor\s*faktur)[\s:]*([A-Z0-9\-/]+)', re.IGNORECASE),
            'invoice_date': re.compile(r'(?:tanggal|date|invoice\s*date)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', re.IGNORECASE),
            'due_date': re.compile(r'(?:due\s*date|jatuh\s*tempo|tanggal\s*jatuh\s*tempo)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', re.IGNORECASE),
            'billing_month': re.compile(r'(?:billing\s*month|bulan\s*tagihan|periode)[\s:]*(\d{4}\d{2}|\d{1,2}[\/\-]\d{4})', re.IGNORECASE),
            
            # Amounts
            'invoice_total': re.compile(r'(?:total|jumlah|amount)[\s:]*(?:rp\.?\s*)?([0-9,\.]+)', re.IGNORECASE),
            'subtotal': re.compile(r'(?:subtotal|sub\s*total)[\s:]*(?:rp\.?\s*)?([0-9,\.]+)', re.IGNORECASE),
            'tax_amount': re.compile(r'(?:ppn|tax|pajak)[\s:]*(?:rp\.?\s*)?([0-9,\.]+)', re.IGNORECASE),
            'tax_percentage': re.compile(r'(?:ppn|tax)[\s:]*(\d+)%', re.IGNORECASE),
            
            # Tax info
            'npwp': re.compile(r'(?:npwp)[\s:]*([0-9\.\-]{15,20})', re.IGNORECASE),
            'faktur_number': re.compile(r'(?:faktur\s*pajak|no\.?\s*faktur)[\s:]*([0-9\-]{3,20})', re.IGNORECASE),
            
            # Bank info
            'bank_name': re.compile(r'(?:bank|nama\s*bank)[\s:]*([A-Z\s]+)', re.IGNORECASE),
            'account_number': re.compile(r'(?:rekening|account|no\.?\s*rek)[\s:]*([0-9\-]{8,20})', re.IGNORECASE),
            'virtual_account': re.compile(r'(?:virtual\s*account|va)[\s:]*([0-9]{10,20})', re.IGNORECASE),
            
            # Names and addresses
            'company_name': re.compile(r'(?:pt\.?\s*|cv\.?\s*)([A-Z\s&,\.]+)', re.IGNORECASE),
            'address': re.compile(r'(?:alamat|address)[\s:]*([A-Za-z0-9\s,\.\-]+)', re.IGNORECASE),
            
            # Line items
            'line_item': re.compile(r'([A-Z0-9\s\-]+)\s+(\d+)\s+([0-9,\.]+)', re.IGNORECASE),
            'bandwidth': re.compile(r'(\d+)\s*mbps', re.IGNORECASE),
            'period': re.compile(r'(\d{6}|\d{4}\d{2})', re.IGNORECASE),
            
            # Signer info
            'signer_name': re.compile(r'(?:ttd|signature|signed\s*by)[\s:]*([A-Z\s\.]+)', re.IGNORECASE),
            'signer_position': re.compile(r'(?:jabatan|position|title)[\s:]*([A-Z\s\.]+)', re.IGNORECASE),
        }
    
    def extract_from_file(self, filepath: str) -> Dict[str, Any]:
        """Extract data from a file (PDF or image)"""
        text = ""
        
        if filepath.lower().endswith('.pdf'):
            if not PDF_AVAILABLE:
                raise Exception("PDF processing not available. Please install pdfplumber.")
            text = self._extract_text_from_pdf(filepath)
        elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            if not TESSERACT_AVAILABLE:
                raise Exception("OCR not available. Please install pytesseract and PIL.")
            text = self._extract_text_from_image(filepath)
        elif filepath.lower().endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise Exception("Unsupported file format")
        
        return self.extract_from_text(text)
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from OCR text"""
        if not text or not text.strip():
            raise Exception("No text provided for extraction")
        
        # Initialize result structure
        result = {
            "invoice_number": "",
            "invoice_date": "",
            "due_date": "",
            "billing_month": "",
            "invoice_total": "",
            "invoice_total_in_words": "",
            "payment_terms": "",
            "vendor": {
                "name": "",
                "tax_id": "",
                "address": ""
            },
            "buyer": {
                "name": "",
                "tax_id": "",
                "address": ""
            },
            "payment": {
                "bank_name": "",
                "virtual_account_number": "",
                "bank_account_number": "",
                "currency": "IDR"
            },
            "signer": {
                "name": "",
                "position": "",
                "signature": ""
            },
            "line_items": [],
            "tax": {
                "subtotal": "",
                "tax_percentage": "",
                "tax_amount": "",
                "luxury_tax_amount": "0",
                "faktur_number": "",
                "tax_issue_date": ""
            }
        }
        
        # Extract basic invoice information with enhanced patterns
        result["invoice_number"] = self._extract_field(text, 'invoice_number') or self._extract_invoice_number_enhanced(text)
        result["invoice_date"] = self._extract_field(text, 'invoice_date')
        result["due_date"] = self._extract_field(text, 'due_date') or self._extract_due_date_enhanced(text)
        result["billing_month"] = self._extract_field(text, 'billing_month') or self._extract_billing_month_enhanced(text)
        result["invoice_total"] = self._extract_field(text, 'invoice_total') or self._extract_total_enhanced(text)
        
        # Extract tax information
        result["tax"]["subtotal"] = self._extract_field(text, 'subtotal')
        result["tax"]["tax_amount"] = self._extract_field(text, 'tax_amount')
        result["tax"]["tax_percentage"] = self._extract_field(text, 'tax_percentage')
        result["tax"]["faktur_number"] = self._extract_field(text, 'faktur_number')
        
        # Extract payment information
        result["payment"]["bank_name"] = self._extract_field(text, 'bank_name')
        result["payment"]["bank_account_number"] = self._extract_field(text, 'account_number')
        result["payment"]["virtual_account_number"] = self._extract_field(text, 'virtual_account')
        
        # Extract vendor and buyer information
        company_names = self._extract_all_matches(text, 'company_name')
        npwp_numbers = self._extract_all_matches(text, 'npwp')
        addresses = self._extract_all_matches(text, 'address')
        
        # Assign first found values to vendor, second to buyer
        if company_names:
            result["vendor"]["name"] = company_names[0]
            if len(company_names) > 1:
                result["buyer"]["name"] = company_names[1]
        
        if npwp_numbers:
            result["vendor"]["tax_id"] = npwp_numbers[0]
            if len(npwp_numbers) > 1:
                result["buyer"]["tax_id"] = npwp_numbers[1]
        
        if addresses:
            result["vendor"]["address"] = addresses[0]
            if len(addresses) > 1:
                result["buyer"]["address"] = addresses[1]
        
        # Extract signer information
        result["signer"]["name"] = self._extract_field(text, 'signer_name')
        result["signer"]["position"] = self._extract_field(text, 'signer_position')
        
        # Extract line items
        result["line_items"] = self._extract_line_items(text)
        
        # Extract payment terms
        if "net 30" in text.lower():
            result["payment_terms"] = "Net 30"
        elif "net 15" in text.lower():
            result["payment_terms"] = "Net 15"
        
        # Extract invoice total in words (terbilang)
        terbilang_pattern = re.compile(r'terbilang[\s:]*([^0-9]+)', re.IGNORECASE)
        match = terbilang_pattern.search(text)
        if match:
            result["invoice_total_in_words"] = match.group(1).strip()
        
        return result
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a single field using regex pattern"""
        pattern = self.patterns.get(field_name)
        if not pattern:
            return ""
        
        match = pattern.search(text)
        return match.group(1).strip() if match else ""
    
    def _extract_all_matches(self, text: str, field_name: str) -> List[str]:
        """Extract all matches for a field"""
        pattern = self.patterns.get(field_name)
        if not pattern:
            return []
        
        matches = pattern.findall(text)
        return [match.strip() for match in matches]
    
    def _extract_line_items(self, text: str) -> List[Dict[str, str]]:
        """Extract line items from invoice text"""
        line_items = []
        
        # Look for ASTINET patterns
        astinet_patterns = [
            r'ASTINET\s*-\s*ONE\s*TIME\s*CHARGES',
            r'ASTINET\s*-\s*MONTHLY\s*RECURRING\s*CHARGES'
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for pattern in astinet_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Look for associated data in nearby lines
                    item = {
                        "description": "",
                        "id": "",
                        "bandwidth": "",
                        "period": "",
                        "amount": "",
                        "type": re.search(pattern, line, re.IGNORECASE).group()
                    }
                    
                    # Search in current and nearby lines for details
                    search_range = lines[max(0, i-2):min(len(lines), i+3)]
                    combined_text = ' '.join(search_range)
                    
                    # Extract ID (numeric pattern)
                    id_match = re.search(r'\b(\d{8,})\b', combined_text)
                    if id_match:
                        item["id"] = id_match.group(1)
                    
                    # Extract bandwidth
                    bandwidth_match = self.patterns['bandwidth'].search(combined_text)
                    if bandwidth_match:
                        item["bandwidth"] = bandwidth_match.group(1) + "MBPS"
                    
                    # Extract period
                    period_match = self.patterns['period'].search(combined_text)
                    if period_match:
                        item["period"] = period_match.group(1)
                    
                    # Extract amount
                    amount_match = re.search(r'\b([0-9,\.]{6,})\b', combined_text)
                    if amount_match:
                        item["amount"] = amount_match.group(1).replace(',', '').replace('.', '')
                    
                    # Extract description (address-like pattern)
                    desc_match = re.search(r'JL\s+[A-Z\s\d]+', combined_text, re.IGNORECASE)
                    if desc_match:
                        item["description"] = desc_match.group().strip()
                    
                    line_items.append(item)
        
        return line_items
    
    def _extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
        return text
    
    def _extract_text_from_image(self, filepath: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image, lang='eng+ind')  # English + Indonesian
            return text
        except Exception as e:
            logging.error(f"Error extracting text from image: {str(e)}")
            raise Exception(f"Failed to extract text from image: {str(e)}")
    
    def _extract_invoice_number_enhanced(self, text: str) -> str:
        """Enhanced invoice number extraction for Telkom format"""
        # Telkom specific patterns
        patterns = [
            r'(?:OFFICIAL\s+RECEIPT\s+NO|Invoice\s+Number)[\s:]*([0-9\-]+)',
            r'Nomor\s+Tagihan[\s.:]*([0-9\-]+)',
            r'(\d{7,}000001[-]\d{6})',  # Telkom format: 4977298000001-202108
            r'(\d{7,}[-]\d{6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_due_date_enhanced(self, text: str) -> str:
        """Enhanced due date extraction"""
        patterns = [
            r'(?:Due\s+Date|Tanggal\s+Akhir\s+Pembayaran)[\s:]*(\d{1,2}\s+\w+\s+\d{4})',
            r'(?:Due\s+Date|Tanggal\s+Akhir\s+Pembayaran)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_billing_month_enhanced(self, text: str) -> str:
        """Enhanced billing month extraction"""
        patterns = [
            r'(?:Billing\s+Month|Bulan\s+Tagihan)[\s:]*(\w+\s+\d{4})',
            r'(?:Billing\s+Month|Bulan\s+Tagihan)[\s:]*(\d{6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_total_enhanced(self, text: str) -> str:
        """Enhanced total amount extraction"""
        patterns = [
            r'(?:Tagihan\s+Bulan\s+Ini|New\s+Charge)[\s\w]*Rp[\s\.]*([0-9,\.]+)',
            r'(?:Total|Jumlah)[\s]*Rp[\s\.]*([0-9,\.]+)',
            r'Rp[\s\.]*([0-9,\.]+)(?=\s*Terbilang)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).replace('.', '').replace(',', '')
                if len(amount) >= 6:  # Reasonable invoice amount
                    return amount
        return ""
