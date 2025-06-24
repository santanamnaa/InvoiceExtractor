#!/usr/bin/env python3
"""
Demo the enhanced extraction with sample OCR text from the Telkom invoice
"""

from enhanced_extraction_service import EnhancedInvoiceExtractor
import json

def demo_sample_text():
    """Test with sample OCR text"""
    
    # Sample OCR text extracted from the Telkom invoice
    sample_text = """
    BIAYA PENGGUNAAN TELKOM Solution
    TELKOM Solution BILLING STATEMENT
    PT. TELKOM INDONESIA (PERSERO) Tbk
    JL JAPATI NO 1 RT 000 RW 000 SADANGSERANG COBLONG KOTA BANDUNG
    NPWP/PKP : 01.000.013.1-093.000
    OFFICIAL RECEIPT NO : 4977298000001-202108
    CV DIGE BATAM
    NPWP. NPWP : 023576879215000
    ANGSANA BATAMINDO BATAM KEP. RIAU 29433
    Nomor Account. Account Number : 4977298
    Bulan Tagihan. Billing Month : Agustus 2021
    Tanggal Akhir Pembayaran. Due Date : 20 Agustus 2021
    
    Tagihan Bulan Ini Rp. 904,839.00
    New Charge
    
    ASTINET - ONE TIME CHARGES Rp. 500,000.00
    ASTINET - MONTHLY RECURRING CHARGES Rp. 322,581.00
    PPN Rp. 82,258.00
    
    Terbilang Sembilan Ratus Empat Ribu Delapan Ratus Tiga Puluh Sembilan Rupiah
    Amount in Words Nine Hundred Four thousand Eight Hundred Thirty Nine Rupiah
    
    BANK MANDIRI Virtual Account No : 88111-8-0004977298
    PT TELKOM DIVRE I SUMATERA (PERSERO) TBK
    No. Rekening : 106.0004651769 (IDR)
    
    Nomor Tagihan. Invoice Number : 4977298000001 - 202108
    NPWP. NPWP : 023576879215000
    Nomor Account. Account Number : 4977298
    Bulan Tagihan. Billing Month : Agustus 2021
    Tanggal Akhir Pembayaran.Due Date : 20 Agustus 2021
    
    No. DESCRIPTION ID BW PERIOD AMOUNT (Rp)
    1 JL PERUMNAS TANJUNG PIAYU BLOK N NOMOR 1537347214 5MBPS 202107 500,000.00
    
    No. DESCRIPTION ID BW PERIOD AMOUNT (Rp)
    1 JL PERUMNAS TANJUNG PIAYU BLOK N NOMOR 1537347214 5MBPS 202107 322,581.00
    
    Faktur Pajak
    Kode dan Nomor Seri Faktur Pajak : 010.007-21.00740493
    Pengusaha Kena Pajak
    Nama : PT. TELKOM INDONESIA (PERSERO) Tbk
    Alamat : JL JAPATI NO 1 RT 000 RW 000 SADANGSERANG COBLONG KOTA BANDUNG
    NPWP : 01.000.013.1-093.000
    Pembeli Barang Kena Pajak / Penerima Jasa Kena Pajak
    Nama : CV DIGE BATAM
    Alamat : ANGSANA BATAMINDO BATAM KEP. RIAU 29433
    NPWP : 02.357.687.9-215.000
    
    Dasar Pengenaan Pajak 822.581,00
    PPN = 10% x Dasar Pengenaan Pajak 82.258,00
    
    KOTA BANDUNG, 01 Agustus 2021
    SANG KOMPIANG MULIARTAWAN
    """
    
    print("Testing Enhanced Extraction with Sample OCR Text")
    print("=" * 50)
    
    # Initialize extractor
    extractor = EnhancedInvoiceExtractor()
    
    # Extract from text
    result = extractor.extract_from_text(sample_text)
    
    print("Extraction Results:")
    print("=" * 30)
    print(f"Invoice Number: {result.get('invoice_number')}")
    print(f"Invoice Date: {result.get('invoice_date')}")
    print(f"Due Date: {result.get('due_date')}")
    print(f"Billing Month: {result.get('billing_month')}")
    print(f"Total Amount: {result.get('invoice_total')}")
    print(f"Vendor Name: {result.get('vendor', {}).get('name')}")
    print(f"Vendor NPWP: {result.get('vendor', {}).get('tax_id')}")
    print(f"Buyer Name: {result.get('buyer', {}).get('name')}")
    print(f"Buyer NPWP: {result.get('buyer', {}).get('tax_id')}")
    print(f"Virtual Account: {result.get('payment', {}).get('virtual_account_number')}")
    print(f"Tax Amount: {result.get('tax', {}).get('tax_amount')}")
    print(f"Faktur Number: {result.get('tax', {}).get('faktur_number')}")
    
    # Save results
    with open('sample_text_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to sample_text_results.json")
    
    return result

if __name__ == "__main__":
    demo_sample_text()