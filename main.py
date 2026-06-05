"""
MEDICAL RECORD OCR & SUMMARIZATION PIPELINE
Using: Ollama (Local Llama 3.1) + Tesseract OCR
✅ 100% FREE - No API costs
✅ 100% Private - Data stays on your machine
✅ Ideal for medical records (HIPAA compatible)
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import json
from datetime import datetime
import cv2
import numpy as np
import ollama
import time

# ==================== CONFIGURATION ====================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OLLAMA_MODEL = "llama3.2:1b"  # Change this line  # or "medllama2", "llama3.1:70b"

# Input PDFs
PHOTO_PDF_PATH = "photo.pdf"
TEXT_PDF_PATH = "Cert Specialists Hospital Recs.pdf"

# Output files
MERGED_PDF = "merged_numbered.pdf"
FINAL_OUTPUT = "final_summary_report.pdf"
LOG_FILE = "processing_log.json"

# ==================== SETUP ====================
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class MedicalRecordProcessor:
    def __init__(self, model_name="llama3.1"):
        self.model = model_name
        self.processing_log = []
        self.start_time = None
        
        # Verify Ollama is working
        print(f"🤖 Testing Ollama connection with model: {model_name}...")
        try:
            test_response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'OK' if you can read this"}]
            )
            print(f"   ✅ Ollama is ready! Response: {test_response['message']['content'][:50]}")
        except Exception as e:
            print(f"   ⚠️ Ollama warning: {e}")
            print("   Make sure Ollama is running and model is pulled")
    
    def preprocess_image_for_ocr(self, image):
        """Enhance image quality for better OCR accuracy"""
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            return Image.fromarray(denoised)
        except:
            return image
    
    def extract_page_as_image(self, pdf_path, page_num, dpi=300):
        """Extract a page from PDF as image"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    
    def add_page_number(self, page, number):
        """Add page numbers"""
        page.insert_text(
            fitz.Point(300, page.rect.height - 30),
            f"- {number} -",
            fontsize=10,
            color=(0.4, 0.4, 0.4)
        )
        page.insert_text(
            fitz.Point(page.rect.width - 50, 20),
            f"Page {number}",
            fontsize=8,
            color=(0.5, 0.5, 0.5)
        )
    
    def merge_pdfs_with_numbers(self, pdf1_path, pdf2_path, output_path):
        """Merge PDFs and add sequential page numbers"""
        print("\n📄 STEP 1: MERGING PDFs")
        
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
        merged = fitz.open()
        page_num = 0
        
        for i in range(len(doc1)):
            page = doc1[i]
            page_num += 1
            self.add_page_number(page, page_num)
            merged.insert_pdf(doc1, from_page=i, to_page=i)
        
        for i in range(len(doc2)):
            page = doc2[i]
            page_num += 1
            self.add_page_number(page, page_num)
            merged.insert_pdf(doc2, from_page=i, to_page=i)
        
        merged.save(output_path)
        
        print(f"   ✅ Merged {page_num} pages with numbers")
        
        doc1.close()
        doc2.close()
        merged.close()
        
        return page_num
    
    def perform_ocr(self, image, page_num):
        """OCR with image preprocessing"""
        try:
            processed_img = self.preprocess_image_for_ocr(image)
            
            # Multiple OCR configs for best results
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 3'
            ]
            
            best_text = ""
            best_words = 0
            
            for config in configs:
                text = pytesseract.image_to_string(processed_img, config=config)
                words = len(text.split())
                if words > best_words:
                    best_text = text
                    best_words = words
            
            return {
                'text': best_text.strip(),
                'word_count': best_words,
                'is_blank': best_words < 10
            }
            
        except Exception as e:
            return {
                'text': f"[OCR Error: {str(e)}]",
                'word_count': 0,
                'is_blank': True
            }
    
    def generate_summary_with_ollama(self, page_text, page_num):
        """Generate medical summary using local Ollama model"""
        if not page_text or len(page_text.strip()) < 20:
            return "No readable content found on this page."
        
        try:
            system_prompt = """You are an expert medical document analyst. Analyze this medical record page carefully.

            Provide a structured summary with:
            1. KEY FINDINGS: Main medical observations
            2. MEDICATIONS: Any drugs or prescriptions mentioned
            3. VITALS: Blood pressure, heart rate, temperature, etc.
            4. ALERTS: Any critical or abnormal findings
            
            Keep it concise (3-4 sentences total). If the text is unclear, state that.
            Only include information actually present in the text."""
            
            # Truncate if needed
            max_chars = 3000
            truncated_text = page_text[:max_chars]
            
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"MEDICAL RECORD - PAGE {page_num}:\n\n{truncated_text}"}
                ],
                options={
                    "temperature": 0.1,  # Low temp for accuracy
                    "num_predict": 300,   # Limit output length
                }
            )
            
            summary = response['message']['content'].strip()
            return summary
            
        except Exception as e:
            print(f"   ⚠️ Ollama error: {str(e)[:100]}")
            return f"Summary unavailable. OCR extracted {len(page_text.split())} words from this page."
    
    def create_output_pdf(self, processed_data, output_path):
        """Create final professional PDF"""
        print("\n📝 STEP 3: CREATING FINAL REPORT")
        
        doc = fitz.open()
        
        for idx, data in enumerate(processed_data):
            page = doc.new_page(width=612, height=792)
            
            # Header
            header_rect = fitz.Rect(40, 30, 572, 55)
            page.draw_rect(header_rect, color=(0.1, 0.4, 0.2), fill=(0.1, 0.4, 0.2))
            page.insert_text(
                fitz.Point(50, 48),
                f"MEDICAL RECORD SUMMARY - Page {data['page_number']}",
                fontsize=13,
                color=(1, 1, 1)
            )
            
            # OCR Section
            page.insert_text(
                fitz.Point(50, 85),
                "📋 EXTRACTED TEXT (OCR):",
                fontsize=11,
                color=(0.1, 0.3, 0.5)
            )
            
            ocr_rect = fitz.Rect(50, 105, 562, 340)
            page.draw_rect(
                fitz.Rect(45, 102, 567, 343),
                color=(0.9, 0.9, 0.9),
                fill=(0.97, 0.97, 0.97)
            )
            page.insert_textbox(
                ocr_rect,
                data['ocr_text'][:2000],
                fontsize=7.5,
                fontname="cour",
                color=(0.2, 0.2, 0.2)
            )
            
            # AI Summary Section
            summary_y = 370
            page.insert_text(
                fitz.Point(50, summary_y),
                "🤖 AI-GENERATED MEDICAL SUMMARY (Ollama):",
                fontsize=11,
                color=(0.2, 0.6, 0.2)
            )
            
            summary_bg = fitz.Rect(45, summary_y+15, 567, 520)
            page.draw_rect(summary_bg, color=(0.2, 0.6, 0.2), fill=(0.95, 1.0, 0.95))
            
            summary_rect = fitz.Rect(55, summary_y+25, 557, 510)
            page.insert_textbox(
                summary_rect,
                data['summary'],
                fontsize=9,
                color=(0.2, 0.2, 0.2)
            )
            
            # Footer
            footer_y = 550
            page.draw_line(
                fitz.Point(50, footer_y),
                fitz.Point(562, footer_y),
                color=(0.7, 0.7, 0.7)
            )
            
            page.insert_text(
                fitz.Point(50, footer_y + 20),
                f"Words: {data['word_count']} | Model: {self.model} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=7,
                color=(0.5, 0.5, 0.5)
            )
        
        doc.save(output_path)
        doc.close()
        print(f"   ✅ Report saved: {output_path}")
    
    def process_medical_record(self, merged_pdf_path, output_pdf_path):
        """Main processing pipeline"""
        print("\n🔍 STEP 2: PROCESSING MEDICAL RECORDS")
        
        doc = fitz.open(merged_pdf_path)
        total_pages = len(doc)
        processed_data = []
        self.start_time = time.time()
        
        print(f"   Pages: {total_pages}")
        print(f"   Model: {self.model}")
        print(f"   OCR: Tesseract\n")
        
        for page_num in range(total_pages):
            page_start = time.time()
            
            # Progress bar
            progress = (page_num + 1) / total_pages * 100
            bar_length = 30
            filled = int(bar_length * (page_num + 1) / total_pages)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r   [{bar}] {progress:.0f}% | Page {page_num + 1}/{total_pages}", end='')
            
            # Extract and process
            page_image = self.extract_page_as_image(merged_pdf_path, page_num)
            ocr_result = self.perform_ocr(page_image, page_num + 1)
            
            # Generate summary
            summary = self.generate_summary_with_ollama(
                ocr_result['text'], 
                page_num + 1
            )
            
            page_data = {
                'page_number': page_num + 1,
                'ocr_text': ocr_result['text'],
                'word_count': ocr_result['word_count'],
                'summary': summary,
                'is_blank': ocr_result['is_blank'],
                'time': round(time.time() - page_start, 2)
            }
            
            processed_data.append(page_data)
        
        print()  # New line after progress bar
        
        # Create final PDF
        self.create_output_pdf(processed_data, output_pdf_path)
        
        # Save log
        with open(LOG_FILE, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        doc.close()
        
        return processed_data

# ==================== MAIN ====================
def main():
    print("\n" + "="*60)
    print("🏥 MEDICAL RECORD PROCESSOR - OLLAMA LOCAL AI")
    print("   ✅ 100% Private | ✅ 100% Free | ✅ Offline")
    print("="*60)
    
    # Check Tesseract
    if not os.path.exists(TESSERACT_PATH):
        print(f"\n❌ Tesseract not found: {TESSERACT_PATH}")
        return
    
    # Check input files
    if not os.path.exists(PHOTO_PDF_PATH):
        print(f"\n❌ Missing: {PHOTO_PDF_PATH}")
        return
    if not os.path.exists(TEXT_PDF_PATH):
        print(f"\n❌ Missing: {TEXT_PDF_PATH}")
        return
    
    # Initialize
    processor = MedicalRecordProcessor(model_name=OLLAMA_MODEL)
    
    try:
        # Step 1: Merge
        total_pages = processor.merge_pdfs_with_numbers(
            PHOTO_PDF_PATH, TEXT_PDF_PATH, MERGED_PDF
        )
        
        # Step 2: Process
        results = processor.process_medical_record(MERGED_PDF, FINAL_OUTPUT)
        
        # Stats
        total_time = time.time() - processor.start_time
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        print(f"   📊 Pages processed: {len(results)}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   📁 Output: {FINAL_OUTPUT}")
        print(f"   🤖 Model: {OLLAMA_MODEL}")
        print("\n✨ Ready for submission!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()