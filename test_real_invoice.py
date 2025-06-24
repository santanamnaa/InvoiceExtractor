#!/usr/bin/env python3
"""
Test the enhanced invoice extraction with the real Telkom invoice
"""

import sys
import os
import json
from enhanced_extraction_service import EnhancedInvoiceExtractor

def test_telkom_invoice():
    """Test extraction on the real Telkom invoice"""
    print("Testing Enhanced Invoice Extraction with Real Telkom Invoice")
    print("=" * 60)
    
    # Initialize enhanced extractor
    try:
        extractor = EnhancedInvoiceExtractor()
        print("✓ Enhanced ML-powered extractor loaded successfully")
        print(f"✓ ML model available: {extractor.ml_model is not None}")
        print(f"✓ Feature extractor available: {extractor.feature_extractor is not None}")
    except Exception as e:
        print(f"❌ Error loading enhanced extractor: {e}")
        return
    
    # Read the PDF file
    pdf_path = "attached_assets/584033305-Telkom-Invoice-4977298-202108_1750763139944.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"\n📄 Processing invoice: {pdf_path}")
    
    try:
        # Extract from PDF
        extracted_data = extractor.extract_from_file(pdf_path)
        
        print("\n🎯 EXTRACTION RESULTS")
        print("=" * 40)
        
        # Display key information
        print(f"📋 Invoice Number: {extracted_data.get('invoice_number', 'Not found')}")
        print(f"📅 Invoice Date: {extracted_data.get('invoice_date', 'Not found')}")
        print(f"📅 Due Date: {extracted_data.get('due_date', 'Not found')}")
        print(f"📅 Billing Month: {extracted_data.get('billing_month', 'Not found')}")
        print(f"💰 Total Amount: {extracted_data.get('invoice_total', 'Not found')}")
        print(f"📝 Amount in Words: {extracted_data.get('invoice_total_in_words', 'Not found')}")
        
        print(f"\n🏢 Vendor Information:")
        vendor = extracted_data.get('vendor', {})
        print(f"   Name: {vendor.get('name', 'Not found')}")
        print(f"   Tax ID (NPWP): {vendor.get('tax_id', 'Not found')}")
        print(f"   Address: {vendor.get('address', 'Not found')}")
        
        print(f"\n🏪 Buyer Information:")
        buyer = extracted_data.get('buyer', {})
        print(f"   Name: {buyer.get('name', 'Not found')}")
        print(f"   Tax ID (NPWP): {buyer.get('tax_id', 'Not found')}")
        print(f"   Address: {buyer.get('address', 'Not found')}")
        
        print(f"\n💳 Payment Information:")
        payment = extracted_data.get('payment', {})
        print(f"   Bank: {payment.get('bank_name', 'Not found')}")
        print(f"   Account Number: {payment.get('bank_account_number', 'Not found')}")
        print(f"   Virtual Account: {payment.get('virtual_account_number', 'Not found')}")
        
        print(f"\n📊 Tax Information:")
        tax = extracted_data.get('tax', {})
        print(f"   Subtotal: {tax.get('subtotal', 'Not found')}")
        print(f"   Tax Amount (PPN): {tax.get('tax_amount', 'Not found')}")
        print(f"   Tax Percentage: {tax.get('tax_percentage', 'Not found')}%")
        print(f"   Faktur Pajak Number: {tax.get('faktur_number', 'Not found')}")
        
        print(f"\n📋 Line Items:")
        line_items = extracted_data.get('line_items', [])
        if line_items:
            for i, item in enumerate(line_items, 1):
                print(f"   {i}. {item.get('type', 'Unknown Type')}")
                print(f"      Description: {item.get('description', 'Not found')}")
                print(f"      ID: {item.get('id', 'Not found')}")
                print(f"      Bandwidth: {item.get('bandwidth', 'Not found')}")
                print(f"      Period: {item.get('period', 'Not found')}")
                print(f"      Amount: {item.get('amount', 'Not found')}")
        else:
            print("   No line items found")
        
        print(f"\n✍️  Signer Information:")
        signer = extracted_data.get('signer', {})
        print(f"   Name: {signer.get('name', 'Not found')}")
        print(f"   Position: {signer.get('position', 'Not found')}")
        
        # Display extraction metadata if available
        metadata = extracted_data.get('extraction_metadata', {})
        if metadata:
            print(f"\n🔍 Extraction Metadata:")
            print(f"   Method: {metadata.get('extraction_method', 'Unknown')}")
            print(f"   Confidence Score: {metadata.get('confidence_score', 0):.2f}")
            print(f"   Extracted Fields: {metadata.get('extracted_fields', 0)}")
            print(f"   Text Length: {metadata.get('text_length', 0)} characters")
        
        # Save results for review
        output_file = "telkom_invoice_extraction_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Full extraction results saved to: {output_file}")
        
        # Test specific expected values from the Telkom invoice
        print(f"\n🎯 VALIDATION AGAINST EXPECTED VALUES")
        print("=" * 45)
        
        expected_values = {
            'Invoice Number': '4977298000001-202108',
            'Total Amount': '904839' or '904,839' or '904.839',
            'Vendor Name': 'TELKOM INDONESIA' or 'PT TELKOM INDONESIA',
            'Buyer Name': 'CV DIGE BATAM',
            'PPN Amount': '82258' or '82,258',
            'Virtual Account': '88111-8-0004977298'
        }
        
        validation_results = []
        
        # Check invoice number
        found_inv = extracted_data.get('invoice_number', '').replace(' ', '').replace('-', '')
        expected_inv = expected_values['Invoice Number'].replace(' ', '').replace('-', '')
        if expected_inv in found_inv or found_inv in expected_inv:
            validation_results.append("✓ Invoice Number: MATCH")
        else:
            validation_results.append(f"❌ Invoice Number: Expected '{expected_values['Invoice Number']}', Found '{extracted_data.get('invoice_number', 'Not found')}'")
        
        # Check total amount
        found_total = extracted_data.get('invoice_total', '').replace(',', '').replace('.', '').replace(' ', '')
        if '904839' in found_total:
            validation_results.append("✓ Total Amount: MATCH")
        else:
            validation_results.append(f"❌ Total Amount: Expected containing '904839', Found '{extracted_data.get('invoice_total', 'Not found')}'")
        
        # Check vendor name
        found_vendor = extracted_data.get('vendor', {}).get('name', '').upper()
        if 'TELKOM' in found_vendor:
            validation_results.append("✓ Vendor Name: MATCH")
        else:
            validation_results.append(f"❌ Vendor Name: Expected containing 'TELKOM', Found '{extracted_data.get('vendor', {}).get('name', 'Not found')}'")
        
        # Check buyer name
        found_buyer = extracted_data.get('buyer', {}).get('name', '').upper()
        if 'DIGE' in found_buyer or 'CV' in found_buyer:
            validation_results.append("✓ Buyer Name: MATCH")
        else:
            validation_results.append(f"❌ Buyer Name: Expected containing 'CV DIGE', Found '{extracted_data.get('buyer', {}).get('name', 'Not found')}'")
        
        for result in validation_results:
            print(f"   {result}")
        
        matches = sum(1 for r in validation_results if r.startswith("✓"))
        total = len(validation_results)
        accuracy = (matches / total) * 100
        
        print(f"\n📊 EXTRACTION ACCURACY: {matches}/{total} ({accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("🎉 EXCELLENT extraction performance!")
        elif accuracy >= 60:
            print("👍 GOOD extraction performance!")
        else:
            print("⚠️  Extraction needs improvement")
        
        return extracted_data
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_telkom_invoice()