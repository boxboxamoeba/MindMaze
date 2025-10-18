# treasure_hunt_gui.py
import sys
import os
import json
import logging
import traceback
import math
import string
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Wayland Compatibility - set early
os.environ['QT_QPA_PLATFORM'] = 'wayland;xcb'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/'

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QLabel, QTextEdit, QListWidget, QListWidgetItem,
                             QPushButton, QTabWidget, QGroupBox, QScrollArea, QFrame,
                             QProgressBar, QFileDialog, QMessageBox, QCheckBox, QComboBox,
                             QLineEdit, QSpinBox, QDoubleSpinBox, QToolBar, QStatusBar,
                             QAction, QTreeWidget, QTreeWidgetItem, QHeaderView, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QDialog, QDialogButtonBox,
                             QSizePolicy, QMenu)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRectF, QEvent, QMimeData, QPoint, pyqtSlot
from PyQt5.QtGui import (QPixmap, QImage, QFont, QPalette, QColor, QIcon, QPen, QBrush, 
                         QDragEnterEvent, QDropEvent, QKeySequence, QPainter, QCursor)
import PyQt5.QtGui as QtGui

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ExifTags
import pytesseract
from pyzbar.pyzbar import decode as pyzbar_decode

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('treasure_hunt_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# BACKEND IMPLEMENTATION (Integrated since import fails)
# =============================================================================

def analyze(file_path: str) -> Dict[str, Any]:
    """Comprehensive file analysis function"""
    result = {
        'file_type': 'unknown',
        'file_size': os.path.getsize(file_path),
        'is_image': False,
        'dimensions': None,
        'exif': {},
        'qr_barcode_outputs': [],
        'stego_candidates': [],
        'text_candidates': [],
        'hidden_unicode': [],
        'entropy': 0.0,
        'metadata': {}
    }
    
    try:
        # Basic file type detection
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']:
            result['file_type'] = 'image'
            result['is_image'] = True
            result.update(analyze_image(file_path))
        elif ext in ['.txt', '.log', '.csv']:
            result['file_type'] = 'text'
            result.update(analyze_text_file(file_path))
        elif ext in ['.pdf']:
            result['file_type'] = 'pdf'
        else:
            result['file_type'] = 'binary'
            result.update(analyze_binary_file(file_path))
            
    except Exception as e:
        logger.error(f"Analysis error for {file_path}: {e}")
        result['error'] = str(e)
        
    return result

def analyze_image(file_path: str) -> Dict[str, Any]:
    """Analyze image file for hidden data"""
    result = {}
    
    try:
        with Image.open(file_path) as img:
            result['dimensions'] = img.size
            
            # EXIF data
            try:
                exif_data = img._getexif()
                if exif_data:
                    result['exif'] = {
                        ExifTags.TAGS.get(tag, tag): value
                        for tag, value in exif_data.items()
                        if isinstance(tag, int) and isinstance(value, (str, int, float, bytes))
                    }
            except:
                result['exif'] = {}
            
            # QR and barcode detection
            cv_image = cv2.imread(file_path)
            if cv_image is not None:
                decoded_objects = pyzbar_decode(cv_image)
                result['qr_barcode_outputs'] = [
                    {
                        'type': obj.type,
                        'data': obj.data.decode('utf-8', errors='ignore'),
                        'rect': obj.rect
                    }
                    for obj in decoded_objects
                ]
            
            # Steganography candidates
            result['stego_candidates'] = detect_stego_candidates(img, file_path)
            
            # Text extraction
            result['text_candidates'] = extract_text_candidates(img)
            
            # Entropy calculation
            result['entropy'] = calculate_image_entropy(img)
            
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        
    return result

def detect_stego_candidates(img: Image.Image, file_path: str) -> List[Dict[str, Any]]:
    """Detect potential steganography in image"""
    candidates = []
    
    try:
        # LSB analysis
        if img.mode in ['RGB', 'RGBA']:
            rgb_data = np.array(img)
            lsb_data = rgb_data & 1
            if np.any(lsb_data):
                candidates.append({
                    'type': 'LSB Steganography',
                    'description': 'Least Significant Bit patterns detected',
                    'confidence': min(0.7, np.mean(lsb_data) * 2)
                })
        
        # File size vs content analysis
        file_size = os.path.getsize(file_path)
        expected_size = img.size[0] * img.size[1] * len(img.mode)
        if file_size > expected_size * 1.2:
            candidates.append({
                'type': 'Appended Data',
                'description': 'File larger than expected for image dimensions',
                'confidence': 0.6
            })
            
    except Exception as e:
        logger.error(f"Stego detection error: {e}")
        
    return candidates

def extract_text_candidates(img: Image.Image) -> List[Dict[str, Any]]:
    """Extract text from image using various methods"""
    candidates = []
    
    try:
        # Method 1: Direct OCR
        try:
            text = pytesseract.image_to_string(img)
            if text.strip():
                score = english_score(text)
                candidates.append({
                    'method': 'Direct OCR',
                    'text': text.strip(),
                    'score': score
                })
        except:
            pass
        
        # Method 2: Preprocessed OCR
        try:
            # Enhance image for better OCR
            enhanced = img.convert('L')
            enhanced = ImageEnhance.Contrast(enhanced).enhance(2.0)
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
            
            text = pytesseract.image_to_string(enhanced)
            if text.strip():
                score = english_score(text)
                candidates.append({
                    'method': 'Enhanced OCR',
                    'text': text.strip(),
                    'score': score
                })
        except:
            pass
            
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates

def calculate_image_entropy(img: Image.Image) -> float:
    """Calculate image entropy"""
    try:
        if img.mode != 'L':
            img = img.convert('L')
        
        histogram = img.histogram()
        histogram_length = sum(histogram)
        
        samples_probability = [float(h) / histogram_length for h in histogram if h != 0]
        return -sum([p * math.log(p, 2) for p in samples_probability])
    except:
        return 0.0

def analyze_text_file(file_path: str) -> Dict[str, Any]:
    """Analyze text file for hidden content"""
    result = {'text_candidates': [], 'hidden_unicode': []}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Basic text analysis
        if content.strip():
            score = english_score(content)
            result['text_candidates'].append({
                'method': 'Direct Text',
                'text': content[:1000] + ('...' if len(content) > 1000 else ''),
                'score': score
            })
            
        # Unicode analysis
        hidden_chars = [c for c in content if ord(c) > 127 and c not in 'àâäèéêëîïôöùûüçÀÂÄÈÉÊËÎÏÔÖÙÛÜÇ']
        if hidden_chars:
            result['hidden_unicode'] = hidden_chars[:20]  # Limit output
            
    except Exception as e:
        logger.error(f"Text file analysis error: {e}")
        
    return result

def analyze_binary_file(file_path: str) -> Dict[str, Any]:
    """Analyze binary file"""
    result = {'entropy': 0.0}
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        # Calculate entropy
        if len(data) > 0:
            byte_counts = Counter(data)
            entropy = 0.0
            total = len(data)
            
            for count in byte_counts.values():
                p = count / total
                entropy -= p * math.log2(p)
                
            result['entropy'] = entropy
            
    except Exception as e:
        logger.error(f"Binary file analysis error: {e}")
        
    return result

# =============================================================================
# CRYPTOGRAPHY FUNCTIONS
# =============================================================================

def atbash(text: Union[str, bytes]) -> str:
    """Atbash cipher implementation"""
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    result = []
    for char in text:
        if char.isalpha():
            if char.isupper():
                result.append(chr(155 - ord(char)))  # 65 + 90 = 155
            else:
                result.append(chr(219 - ord(char)))  # 97 + 122 = 219
        else:
            result.append(char)
    return ''.join(result)

def vigenere_decrypt(text: str, key: str) -> str:
    """Vigenère cipher decryption"""
    result = []
    key = key.upper()
    key_index = 0
    
    for char in text:
        if char.isalpha():
            # Determine the shift from the key
            shift = ord(key[key_index % len(key)]) - ord('A')
            
            if char.isupper():
                base = ord('A')
                result.append(chr((ord(char) - base - shift) % 26 + base))
            else:
                base = ord('a')
                result.append(chr((ord(char) - base - shift) % 26 + base))
            
            key_index += 1
        else:
            result.append(char)
    
    return ''.join(result)

def english_score(text: str) -> float:
    """Score how English-like the text is"""
    if not text:
        return 0.0
    
    # Common English words (expanded list)
    common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 
                   'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 
                   'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 
                   'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 
                   'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 
                   'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make',
                   'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take'}
    
    text_lower = text.lower()
    words = text_lower.split()
    
    if not words:
        return 0.0
    
    # Count common words
    common_count = sum(1 for word in words if word in common_words)
    word_score = common_count / len(words)
    
    # Character frequency score
    char_freq = Counter(text_lower)
    total_chars = sum(char_freq.values())
    
    if total_chars == 0:
        return 0.0
    
    english_freq = {'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7}
    freq_score = 0.0
    
    for char, expected_freq in english_freq.items():
        actual_freq = (char_freq.get(char, 0) / total_chars) * 100
        freq_score += 1.0 - min(1.0, abs(actual_freq - expected_freq) / expected_freq)
    
    freq_score /= len(english_freq)
    
    # Combined score (weighted towards common words)
    return (word_score * 0.7 + freq_score * 0.3)

def rail_fence_decrypt(text: str, rails: int) -> str:
    """Rail fence cipher decryption"""
    if rails <= 1:
        return text
    
    # Create the fence pattern
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1
    
    # Mark positions
    for i in range(len(text)):
        fence[rail].append(i)
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction = -direction
    
    # Flatten the fence pattern
    indices = []
    for rail in fence:
        indices.extend(rail)
    
    # Reconstruct the text
    result = [''] * len(text)
    for i, char in zip(indices, text):
        result[i] = char
    
    return ''.join(result)

def xor_decrypt(data: bytes, key: bytes) -> bytes:
    """XOR decryption"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    if isinstance(key, str):
        key = key.encode('utf-8')
    
    result = bytearray()
    key_length = len(key)
    
    for i, byte in enumerate(data):
        result.append(byte ^ key[i % key_length])
    
    return bytes(result)

COMMON_KEYS = ["key", "password", "secret", "treasure", "hidden", "cipher", "code", "crypto", "hunt", "quest"]

# =============================================================================
# GUI CLASSES (Enhanced and Fixed)
# =============================================================================

class ProcessingThread(QThread):
    """Thread for running analysis to keep UI responsive"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._is_running = True
    
    def run(self):
        try:
            if not self._is_running:
                return
                
            self.progress_signal.emit(10, "Starting analysis...")
            
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            if not self._is_running:
                return
                
            self.progress_signal.emit(30, "Analyzing file structure...")
            result = analyze(self.file_path)
            
            if not self._is_running:
                return
                
            self.progress_signal.emit(70, "Processing results...")
            # Add additional processing here if needed
            
            if not self._is_running:
                return
                
            self.progress_signal.emit(100, "Analysis complete")
            self.finished_signal.emit(result)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            self.error_signal.emit(f"Analysis failed: {str(e)}")
    
    def stop(self):
        """Safely stop the thread"""
        self._is_running = False
        self.quit()
        self.wait(5000)

class ImageViewer(QGraphicsView):
    """Custom graphics view for image display with zoom and pan"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.zoom_factor = 1.0
        self.current_pixmap = None
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
    def set_image(self, pixmap: QPixmap):
        """Set the image to display"""
        self.scene.clear()
        self.current_pixmap = pixmap
        if pixmap and not pixmap.isNull():
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.zoom_factor = 1.0
        else:
            logger.warning("Attempted to set null or invalid pixmap")
    
    def wheelEvent(self, event):
        """Handle zoom with mouse wheel"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        
        old_pos = self.mapToScene(event.pos())
        
        if event.angleDelta().y() > 0:
            if self.zoom_factor < self.max_zoom:
                self.scale(zoom_in_factor, zoom_in_factor)
                self.zoom_factor *= zoom_in_factor
        else:
            if self.zoom_factor > self.min_zoom:
                self.scale(zoom_out_factor, zoom_out_factor)
                self.zoom_factor *= zoom_out_factor
        
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    def reset_zoom(self):
        """Reset zoom to fit view"""
        if self.scene.items():
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.zoom_factor = 1.0

class TextDetectionDialog(QDialog):
    """Dialog for OCR configuration and results"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OCR Text Detection")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Configuration section
        config_group = QGroupBox("OCR Configuration")
        config_layout = QHBoxLayout(config_group)
        
        config_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["eng", "fra", "deu", "spa", "ita", "por", "rus"])
        config_layout.addWidget(self.lang_combo)
        
        config_layout.addWidget(QLabel("OCR Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Default", "LSTM", "Legacy"])
        config_layout.addWidget(self.engine_combo)
        
        config_layout.addStretch()
        
        self.retry_btn = QPushButton("Retry OCR")
        self.retry_btn.clicked.connect(self.retry_ocr)
        config_layout.addWidget(self.retry_btn)
        
        layout.addWidget(config_group)
        
        # Results section
        results_group = QGroupBox("Detected Text")
        results_layout = QVBoxLayout(results_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Detected text will appear here...")
        results_layout.addWidget(self.text_edit)
        
        # Confidence display
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Score:"))
        self.confidence_label = QLabel("N/A")
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addStretch()
        results_layout.addLayout(confidence_layout)
        
        layout.addWidget(results_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.current_image = None
    
    def set_image(self, image: Image.Image):
        """Set image for OCR analysis"""
        self.current_image = image
        # Auto-run OCR when image is set
        QTimer.singleShot(100, self.retry_ocr)
    
    def retry_ocr(self):
        """Retry OCR with current settings"""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "No image available for OCR")
            return
            
        try:
            self.retry_btn.setEnabled(False)
            self.retry_btn.setText("Processing...")
            
            # Configure OCR based on selections
            config = ""
            if self.engine_combo.currentText() == "LSTM":
                config += "--oem 1"
            elif self.engine_combo.currentText() == "Legacy":
                config += "--oem 0"
            
            lang = self.lang_combo.currentText()
            
            # Perform OCR
            text = pytesseract.image_to_string(self.current_image, lang=lang, config=config)
            self.text_edit.setPlainText(text)
            
            # Get confidence
            try:
                data = pytesseract.image_to_data(self.current_image, lang=lang, output_type=pytesseract.Output.DICT)
                confidences = [float(conf) for conf in data['conf'] if float(conf) > 0]
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    self.confidence_label.setText(f"{avg_conf:.1f}%")
                else:
                    self.confidence_label.setText("N/A")
            except Exception as e:
                logger.warning(f"Could not calculate confidence: {e}")
                self.confidence_label.setText("N/A")
                
        except Exception as e:
            QMessageBox.warning(self, "OCR Error", f"Failed to perform OCR: {str(e)}")
            logger.error(f"OCR error: {e}")
        finally:
            self.retry_btn.setEnabled(True)
            self.retry_btn.setText("Retry OCR")

class CryptoAnalysisDialog(QDialog):
    """Dialog for cryptographic analysis tools"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cryptographic Analysis Tools")
        self.setModal(True)
        self.resize(900, 700)
        
        layout = QVBoxLayout(self)
        
        # Input section
        input_group = QGroupBox("Input Text")
        input_layout = QVBoxLayout(input_group)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text to analyze here...")
        input_layout.addWidget(self.input_text)
        
        layout.addWidget(input_group)
        
        # Tools section
        tools_tabs = QTabWidget()
        
        # Frequency Analysis tab
        freq_tab = QWidget()
        freq_layout = QVBoxLayout(freq_tab)
        
        freq_controls = QHBoxLayout()
        freq_controls.addWidget(QLabel("Analysis Type:"))
        self.freq_type = QComboBox()
        self.freq_type.addItems(["Single Character", "Bigram", "Trigram"])
        freq_controls.addWidget(self.freq_type)
        
        self.analyze_freq_btn = QPushButton("Analyze Frequency")
        self.analyze_freq_btn.clicked.connect(self.analyze_frequency)
        freq_controls.addWidget(self.analyze_freq_btn)
        freq_controls.addStretch()
        
        freq_layout.addLayout(freq_controls)
        
        self.freq_results = QTextEdit()
        self.freq_results.setReadOnly(True)
        freq_layout.addWidget(self.freq_results)
        
        tools_tabs.addTab(freq_tab, "Frequency Analysis")
        
        # Substitution Cipher tab
        sub_tab = QWidget()
        sub_layout = QVBoxLayout(sub_tab)
        
        sub_controls = QHBoxLayout()
        sub_controls.addWidget(QLabel("Cipher Type:"))
        self.cipher_type = QComboBox()
        self.cipher_type.addItems(["Caesar", "Atbash", "Vigenère", "Custom Substitution"])
        sub_controls.addWidget(self.cipher_type)
        
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Key (for Vigenère)")
        sub_controls.addWidget(self.key_input)
        
        self.decrypt_btn = QPushButton("Decrypt")
        self.decrypt_btn.clicked.connect(self.decrypt_cipher)
        sub_controls.addWidget(self.decrypt_btn)
        sub_controls.addStretch()
        
        sub_layout.addLayout(sub_controls)
        
        self.decrypt_results = QTextEdit()
        self.decrypt_results.setReadOnly(True)
        sub_layout.addWidget(self.decrypt_results)
        
        tools_tabs.addTab(sub_tab, "Substitution Ciphers")
        
        # Brute Force tab
        brute_tab = QWidget()
        brute_layout = QVBoxLayout(brute_tab)
        
        brute_controls = QHBoxLayout()
        brute_controls.addWidget(QLabel("Attack Type:"))
        self.attack_type = QComboBox()
        self.attack_type.addItems(["Caesar (all shifts)", "Rail Fence (2-10 rails)", "XOR (common keys)"])
        brute_controls.addWidget(self.attack_type)
        
        self.brute_force_btn = QPushButton("Brute Force")
        self.brute_force_btn.clicked.connect(self.brute_force)
        brute_controls.addWidget(self.brute_force_btn)
        brute_controls.addStretch()
        
        brute_layout.addLayout(brute_controls)
        
        self.brute_results = QTextEdit()
        self.brute_results.setReadOnly(True)
        brute_layout.addWidget(self.brute_results)
        
        tools_tabs.addTab(brute_tab, "Brute Force")
        
        layout.addWidget(tools_tabs)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
    
    def analyze_frequency(self):
        """Perform frequency analysis on input text"""
        text = self.input_text.toPlainText().upper()
        if not text:
            self.freq_results.setPlainText("Please enter text to analyze.")
            return
        
        analysis_type = self.freq_type.currentText()
        
        try:
            if analysis_type == "Single Character":
                from collections import Counter
                # Count only alphabetic characters
                counter = Counter(c for c in text if c.isalpha())
                total = sum(counter.values())
                
                if total == 0:
                    self.freq_results.setPlainText("No alphabetic characters found for frequency analysis.")
                    return
                
                result = "Character Frequency Analysis:\n\n"
                result += "Char  Count  Frequency\n"
                result += "-" * 30 + "\n"
                
                for char, count in counter.most_common():
                    freq = (count / total) * 100
                    result += f"{char:^5} {count:^6} {freq:>6.2f}%\n"
                
                # Compare to English frequency
                english_freq = {'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 
                               'N': 6.7, 'S': 6.3, 'H': 6.1, 'R': 6.0, 'D': 4.3}
                
                result += "\nComparison to English frequencies:\n"
                for char in 'ETAOINSHRD':
                    actual = (counter[char] / total * 100) if char in counter else 0
                    expected = english_freq.get(char, 0)
                    diff = actual - expected
                    result += f"{char}: {actual:5.2f}% (expected {expected:4.1f}%, diff: {diff:6.2f}%)\n"
                    
            self.freq_results.setPlainText(result)
            
        except Exception as e:
            self.freq_results.setPlainText(f"Error during frequency analysis: {str(e)}")
            logger.error(f"Frequency analysis error: {e}")
    
    def decrypt_cipher(self):
        """Decrypt using selected cipher"""
        text = self.input_text.toPlainText()
        cipher_type = self.cipher_type.currentText()
        key = self.key_input.text()
        
        if not text:
            self.decrypt_results.setPlainText("Please enter text to decrypt.")
            return
        
        try:
            result = ""
            
            if cipher_type == "Caesar":
                # Try all Caesar shifts
                result = "Caesar Cipher Decryption (all shifts):\n\n"
                for shift in range(1, 26):
                    decrypted = ""
                    for char in text:
                        if char.isalpha():
                            base = ord('A') if char.isupper() else ord('a')
                            decrypted += chr((ord(char) - base - shift) % 26 + base)
                        else:
                            decrypted += char
                    score = english_score(decrypted)
                    result += f"Shift {shift:2} (score: {score:.3f}): {decrypted}\n"
                    
            elif cipher_type == "Atbash":
                result = "Atbash Decryption:\n\n"
                decrypted = atbash(text)
                score = english_score(decrypted)
                result += f"Score: {score:.3f}\n{decrypted}"
                
            elif cipher_type == "Vigenère":
                if not key:
                    QMessageBox.warning(self, "Missing Key", "Please enter a Vigenère key")
                    return
                result = f"Vigenère Decryption (key: {key}):\n\n"
                decrypted = vigenere_decrypt(text, key)
                score = english_score(decrypted)
                result += f"Score: {score:.3f}\n{decrypted}"
            
            self.decrypt_results.setPlainText(result)
            
        except Exception as e:
            self.decrypt_results.setPlainText(f"Error during decryption: {str(e)}")
            logger.error(f"Decryption error: {e}")
    
    def brute_force(self):
        """Perform brute force attack"""
        text = self.input_text.toPlainText()
        attack_type = self.attack_type.currentText()
        
        if not text:
            self.brute_results.setPlainText("Please enter text to brute force.")
            return
        
        try:
            result = f"Brute Force Results ({attack_type}):\n\n"
            
            if attack_type == "Caesar (all shifts)":
                for shift in range(1, 26):
                    decrypted = ""
                    for char in text:
                        if char.isalpha():
                            base = ord('A') if char.isupper() else ord('a')
                            decrypted += chr((ord(char) - base - shift) % 26 + base)
                        else:
                            decrypted += char
                    score = english_score(decrypted)
                    if score > 0.1:
                        result += f"Shift {shift:2} (score: {score:.3f}): {decrypted}\n"
                        
            elif attack_type == "Rail Fence (2-10 rails)":
                for rails in range(2, 11):
                    decrypted = rail_fence_decrypt(text, rails)
                    score = english_score(decrypted)
                    if score > 0.1:
                        result += f"Rails {rails} (score: {score:.3f}): {decrypted}\n"
                        
            elif attack_type == "XOR (common keys)":
                text_bytes = text.encode()
                for key in COMMON_KEYS:
                    decrypted_bytes = xor_decrypt(text_bytes, key.encode())
                    try:
                        decrypted = decrypted_bytes.decode('utf-8', errors='ignore')
                        score = english_score(decrypted)
                        if score > 0.1:
                            result += f"XOR key '{key}' (score: {score:.3f}): {decrypted}\n"
                    except:
                        pass
            
            self.brute_results.setPlainText(result)
            
        except Exception as e:
            self.brute_results.setPlainText(f"Error during brute force: {str(e)}")
            logger.error(f"Brute force error: {e}")

class AISuggestionWidget(QWidget):
    """Widget for displaying AI suggestions and hints"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.suggestion_label = QLabel("AI Suggestions")
        self.suggestion_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.suggestion_label)
        
        self.suggestion_text = QTextEdit()
        self.suggestion_text.setReadOnly(True)
        self.suggestion_text.setPlaceholderText("AI suggestions will appear here after analysis...")
        layout.addWidget(self.suggestion_text)
        
        self.next_steps_label = QLabel("Recommended Next Steps")
        self.next_steps_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.next_steps_label)
        
        self.next_steps_list = QListWidget()
        layout.addWidget(self.next_steps_list)
        
        self.manual_hint_input = QLineEdit()
        self.manual_hint_input.setPlaceholderText("Enter manual hint or clue...")
        layout.addWidget(self.manual_hint_input)
        
        self.add_hint_btn = QPushButton("Add Manual Hint")
        self.add_hint_btn.clicked.connect(self.add_manual_hint)
        layout.addWidget(self.add_hint_btn)
    
    def update_suggestions(self, analysis_result: Dict[str, Any]):
        """Update AI suggestions based on analysis results"""
        suggestions = []
        next_steps = []
        
        # Generate suggestions based on analysis
        if analysis_result.get('is_image', False):
            if analysis_result.get('exif'):
                suggestions.append("EXIF data found - check for hidden metadata")
                next_steps.append("Review EXIF metadata for clues")
            
            if analysis_result.get('qr_barcode_outputs'):
                suggestions.append("QR/Barcodes detected - review decoded content")
                next_steps.append("Verify QR/barcode decoding results")
            
            if analysis_result.get('stego_candidates'):
                suggestions.append("Potential steganography detected - investigate LSB patterns")
                next_steps.append("Analyze steganography candidates")
        
        if analysis_result.get('text_candidates'):
            best_candidate = analysis_result['text_candidates'][0] if analysis_result['text_candidates'] else {}
            if best_candidate.get('score', 0) > 0.5:
                suggestions.append(f"High-confidence text candidate found: {best_candidate.get('method', 'unknown')}")
            next_steps.append("Review text decoding candidates")
        
        if analysis_result.get('hidden_unicode'):
            suggestions.append("Hidden Unicode characters detected - investigate for steganography")
            next_steps.append("Analyze hidden Unicode characters")
        
        if analysis_result.get('entropy', 0) > 7.5:
            suggestions.append("High entropy detected - possible encryption or compression")
            next_steps.append("Investigate high entropy regions")
        
        # Update UI
        self.suggestion_text.setPlainText("\n".join(suggestions) if suggestions else "No specific suggestions at this time.")
        
        self.next_steps_list.clear()
        for step in next_steps:
            self.next_steps_list.addItem(step)
    
    def add_manual_hint(self):
        """Add a manual hint to the suggestions"""
        hint = self.manual_hint_input.text().strip()
        if hint:
            current_text = self.suggestion_text.toPlainText()
            new_text = f"{current_text}\n\n[Manual Hint] {hint}" if current_text else f"[Manual Hint] {hint}"
            self.suggestion_text.setPlainText(new_text)
            self.manual_hint_input.clear()

class TreasureHuntGUI(QMainWindow):
    """Main GUI window for the enhanced treasure hunt pipeline"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.analysis_result = None
        self.current_image = None
        self.current_filtered_image = None
        self.current_image_variants = {}
        self.analysis_thread = None
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Treasure Hunt Analysis Suite")
        self.setGeometry(100, 50, 1600, 1000)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #88c0d0;
            }
            QPushButton {
                background-color: #4c566a;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5e81ac;
            }
            QPushButton:disabled {
                background-color: #3b4252;
                color: #666;
            }
            QTextEdit, QListWidget, QTreeWidget {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 3px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - file browser and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Center panel - image display and analysis
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # Right panel - results and AI suggestions
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([300, 700, 400])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create toolbar
        self.create_toolbar()
    
    def create_left_panel(self):
        """Create the left panel with file browser and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File browser section
        file_group = QGroupBox("File Browser")
        file_layout = QVBoxLayout(file_group)
        
        # File controls
        file_controls = QHBoxLayout()
        self.browse_btn = QPushButton("Browse Files")
        file_controls.addWidget(self.browse_btn)
        
        self.browse_folder_btn = QPushButton("Browse Folder")
        file_controls.addWidget(self.browse_folder_btn)
        
        file_controls.addStretch()
        
        file_layout.addLayout(file_controls)
        
        # File list
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)
        
        # Current file info
        self.file_info = QTextEdit()
        self.file_info.setMaximumHeight(150)
        self.file_info.setReadOnly(True)
        file_layout.addWidget(self.file_info)
        
        layout.addWidget(file_group)
        
        # Processing controls
        process_group = QGroupBox("Processing Controls")
        process_layout = QVBoxLayout(process_group)
        
        self.analyze_btn = QPushButton("Analyze File")
        self.analyze_btn.setEnabled(False)
        process_layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        process_layout.addWidget(self.progress_label)
        
        # Image processing filters
        filters_group = QGroupBox("Image Filters")
        filters_layout = QVBoxLayout(filters_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "Original", "Grayscale", "Sharpen", "Upscale", "CLAHE", "Invert",
            "Edge Enhance", "Find Edges", "Contour", "Emboss", "Enhance Contrast"
        ])
        filters_layout.addWidget(self.filter_combo)
        
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.setEnabled(False)
        filters_layout.addWidget(self.apply_filter_btn)
        
        process_layout.addWidget(filters_group)
        
        layout.addWidget(process_group)
        layout.addStretch()
        
        return panel
    
    def create_center_panel(self):
        """Create the center panel with image display"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image display tabs
        self.image_tabs = QTabWidget()
        
        # Original image tab
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_viewer = ImageViewer()
        original_layout.addWidget(self.original_viewer)
        self.image_tabs.addTab(self.original_tab, "Original")
        
        # Processed image tab
        self.processed_tab = QWidget()
        processed_layout = QVBoxLayout(self.processed_tab)
        self.processed_viewer = ImageViewer()
        processed_layout.addWidget(self.processed_viewer)
        self.image_tabs.addTab(self.processed_tab, "Processed")
        
        # Comparison tab
        self.comparison_tab = QWidget()
        comparison_layout = QHBoxLayout(self.comparison_tab)
        self.comparison_viewer1 = ImageViewer()
        self.comparison_viewer2 = ImageViewer()
        comparison_layout.addWidget(self.comparison_viewer1)
        comparison_layout.addWidget(self.comparison_viewer2)
        self.image_tabs.addTab(self.comparison_tab, "Comparison")
        
        layout.addWidget(self.image_tabs)
        
        # Image controls
        controls_layout = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("Zoom In")
        controls_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        controls_layout.addWidget(self.zoom_out_btn)
        
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        controls_layout.addWidget(self.reset_zoom_btn)
        
        controls_layout.addStretch()
        
        self.ocr_btn = QPushButton("OCR Text Detection")
        controls_layout.addWidget(self.ocr_btn)
        
        self.crypto_btn = QPushButton("Crypto Tools")
        controls_layout.addWidget(self.crypto_btn)
        
        layout.addLayout(controls_layout)
        
        return panel
    
    def create_right_panel(self):
        """Create the right panel with results and AI suggestions"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Analysis results tab
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Type", "Details", "Confidence"])
        self.results_tree.setColumnWidth(0, 150)
        self.results_tree.setColumnWidth(1, 300)
        analysis_layout.addWidget(self.results_tree)
        
        self.results_tabs.addTab(self.analysis_tab, "Analysis Results")
        
        # Text candidates tab
        self.text_tab = QWidget()
        text_layout = QVBoxLayout(self.text_tab)
        
        self.text_candidates_list = QListWidget()
        text_layout.addWidget(self.text_candidates_list)
        
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setMaximumHeight(200)
        text_layout.addWidget(self.text_preview)
        
        self.results_tabs.addTab(self.text_tab, "Text Candidates")
        
        # Logs tab
        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout(self.logs_tab)
        
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        logs_layout.addWidget(self.logs_text)
        
        self.results_tabs.addTab(self.logs_tab, "Processing Logs")
        
        layout.addWidget(self.results_tabs)
        
        # AI suggestions widget
        self.ai_widget = AISuggestionWidget()
        layout.addWidget(self.ai_widget)
        
        return panel
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open File', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.browse_files)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.triggered.connect(self.browse_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export Report', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        ocr_action = QAction('OCR Settings', self)
        ocr_action.triggered.connect(self.show_ocr_dialog)
        tools_menu.addAction(ocr_action)
        
        crypto_action = QAction('Cryptographic Tools', self)
        crypto_action.triggered.connect(self.show_crypto_dialog)
        tools_menu.addAction(crypto_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        reset_zoom_action = QAction('Reset Zoom', self)
        reset_zoom_action.setShortcut('Ctrl+0')
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)
    
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        open_action = QAction("Open File", self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.browse_files)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        analyze_action = QAction("Analyze", self)
        analyze_action.triggered.connect(self.analyze_file)
        toolbar.addAction(analyze_action)
        
        toolbar.addSeparator()
        
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_report)
        toolbar.addAction(export_action)
    
    def setup_connections(self):
        """Setup signal connections"""
        # File browser
        self.browse_btn.clicked.connect(self.browse_files)
        self.browse_folder_btn.clicked.connect(self.browse_folder)
        self.file_list.itemClicked.connect(self.on_file_selected)
        
        # Analysis
        self.analyze_btn.clicked.connect(self.analyze_file)
        
        # Image controls
        self.apply_filter_btn.clicked.connect(self.apply_image_filter)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        
        # Tools
        self.ocr_btn.clicked.connect(self.show_ocr_dialog)
        self.crypto_btn.clicked.connect(self.show_crypto_dialog)
        
        # Text candidates
        self.text_candidates_list.itemClicked.connect(self.on_text_candidate_selected)
    
    def browse_files(self):
        """Browse for files to analyze"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Analyze",
            "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.bmp *.tiff);;Text Files (*.txt);;PDF Files (*.pdf)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def browse_folder(self):
        """Browse for folders containing files to analyze"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Analyze"
        )
        
        if folder_path:
            self.load_folder(folder_path)
    
    def load_file(self, file_path: str):
        """Load a single file for analysis"""
        self.current_file = file_path
        self.file_list.clear()
        self.file_list.addItem(QListWidgetItem(file_path))
        
        # Update file info
        file_info = f"File: {os.path.basename(file_path)}\n"
        file_info += f"Size: {os.path.getsize(file_path)} bytes\n"
        file_info += f"Type: {os.path.splitext(file_path)[1]}\n"
        file_info += f"Modified: {datetime.fromtimestamp(os.path.getmtime(file_path))}"
        
        self.file_info.setPlainText(file_info)
        self.analyze_btn.setEnabled(True)
        
        # Clear previous results
        self.clear_results()
        
        # Load image preview if it's an image
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            self.load_image_preview(file_path)
        else:
            self.original_viewer.set_image(QPixmap())
            self.processed_viewer.set_image(QPixmap())
            self.apply_filter_btn.setEnabled(False)
    
    def load_folder(self, folder_path: str):
        """Load all files from a folder"""
        self.file_list.clear()
        
        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    self.file_list.addItem(QListWidgetItem(file_path))
            
            if self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)
                self.on_file_selected(self.file_list.item(0))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load folder: {str(e)}")
            logger.error(f"Folder loading error: {e}")
    
    def on_file_selected(self, item):
        """Handle file selection from list"""
        file_path = item.text()
        self.load_file(file_path)
    
    def load_image_preview(self, file_path: str):
        """Load image preview"""
        try:
            # Load original image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.original_viewer.set_image(pixmap)
                self.current_image = Image.open(file_path)
                self.current_filtered_image = self.current_image.copy()
                
                # Update processed viewer with original
                qimage = self.pil_to_qimage(self.current_filtered_image)
                self.processed_viewer.set_image(QPixmap.fromImage(qimage))
                
                self.apply_filter_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Could not load image")
                logger.error(f"Could not load image: {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load image: {str(e)}")
            logger.error(f"Image loading error: {e}")
    
    def analyze_file(self):
        """Start file analysis in a separate thread"""
        if not self.current_file:
            QMessageBox.warning(self, "No File", "Please select a file first")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")
        
        # Disable analyze button during processing
        self.analyze_btn.setEnabled(False)
        
        # Start analysis thread
        self.analysis_thread = ProcessingThread(self.current_file)
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.finished_signal.connect(self.analysis_finished)
        self.analysis_thread.error_signal.connect(self.analysis_error)
        self.analysis_thread.start()
        
        # Add to logs
        self.log_message(f"Started analysis of: {self.current_file}")
    
    def update_progress(self, value: int, message: str):
        """Update progress bar and label"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def analysis_finished(self, result: Dict[str, Any]):
        """Handle analysis completion"""
        self.analysis_result = result
        
        # Update progress
        self.progress_bar.setValue(100)
        self.progress_label.setText("Analysis complete")
        
        # Re-enable analyze button
        self.analyze_btn.setEnabled(True)
        
        # Display results
        self.display_results(result)
        
        # Update AI suggestions
        self.ai_widget.update_suggestions(result)
        
        # Add to logs
        self.log_message("Analysis completed successfully")
        
        # Hide progress after delay
        QTimer.singleShot(2000, self.hide_progress)
    
    def analysis_error(self, error_message: str):
        """Handle analysis errors"""
        QMessageBox.critical(self, "Analysis Error", error_message)
        
        # Reset UI
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # Add to logs
        self.log_message(f"Analysis failed: {error_message}")
    
    def hide_progress(self):
        """Hide progress indicators"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
    
    def display_results(self, result: Dict[str, Any]):
        """Display analysis results in the tree widget"""
        self.results_tree.clear()
        
        # File information
        file_info = QTreeWidgetItem(["File Info", "", ""])
        file_info.addChild(QTreeWidgetItem(["File Type", str(result.get('file_type', 'Unknown')), ""]))
        file_info.addChild(QTreeWidgetItem(["File Size", str(result.get('file_size', 0)), ""]))
        file_info.addChild(QTreeWidgetItem(["Entropy", f"{result.get('entropy', 0):.3f}", ""]))
        self.results_tree.addTopLevelItem(file_info)
        
        # Image analysis
        if result.get('is_image', False):
            image_info = QTreeWidgetItem(["Image Analysis", "", ""])
            
            if 'dimensions' in result:
                image_info.addChild(QTreeWidgetItem(["Dimensions", str(result['dimensions']), ""]))
            
            if 'exif' in result and result['exif']:
                exif_item = QTreeWidgetItem(["EXIF Data", "Available", ""])
                for key, value in list(result['exif'].items())[:10]:  # Limit to first 10 items
                    exif_item.addChild(QTreeWidgetItem([str(key), str(value), ""]))
                image_info.addChild(exif_item)
            
            if 'qr_barcode_outputs' in result and result['qr_barcode_outputs']:
                qr_item = QTreeWidgetItem(["QR/Barcodes", f"{len(result['qr_barcode_outputs'])} found", ""])
                for i, output in enumerate(result['qr_barcode_outputs']):
                    qr_item.addChild(QTreeWidgetItem([f"Code {i+1}", str(output.get('data', '')), ""]))
                image_info.addChild(qr_item)
            
            if 'stego_candidates' in result and result['stego_candidates']:
                stego_item = QTreeWidgetItem(["Steganography", f"{len(result['stego_candidates'])} candidates", ""])
                for candidate in result['stego_candidates']:
                    stego_item.addChild(QTreeWidgetItem([candidate.get('type', 'Unknown'), 
                                                       candidate.get('description', ''), 
                                                       f"{candidate.get('confidence', 0):.2f}"]))
                image_info.addChild(stego_item)
            
            self.results_tree.addTopLevelItem(image_info)
        
        # Text analysis
        text_info = QTreeWidgetItem(["Text Analysis", "", ""])
        
        if 'text_candidates' in result and result['text_candidates']:
            for candidate in result['text_candidates']:
                text_info.addChild(QTreeWidgetItem([
                    candidate.get('method', 'Unknown'),
                    candidate.get('text', '')[:100] + "..." if len(candidate.get('text', '')) > 100 else candidate.get('text', ''),
                    f"{candidate.get('score', 0):.2f}"
                ]))
        
        if 'hidden_unicode' in result and result['hidden_unicode']:
            text_info.addChild(QTreeWidgetItem(["Hidden Unicode", f"{len(result['hidden_unicode'])} characters", ""]))
        
        self.results_tree.addTopLevelItem(text_info)
        
        # Expand all items
        self.results_tree.expandAll()
        
        # Update text candidates list
        self.text_candidates_list.clear()
        if 'text_candidates' in result:
            for i, candidate in enumerate(result['text_candidates']):
                item = QListWidgetItem(f"{candidate.get('method', 'Unknown')} (score: {candidate.get('score', 0):.2f})")
                item.setData(Qt.UserRole, candidate)
                self.text_candidates_list.addItem(item)
    
    def on_text_candidate_selected(self, item):
        """Handle text candidate selection"""
        candidate = item.data(Qt.UserRole)
        if candidate and 'text' in candidate:
            self.text_preview.setPlainText(candidate['text'])
    
    def apply_image_filter(self):
        """Apply selected image filter"""
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return
        
        filter_name = self.filter_combo.currentText()
        
        try:
            filtered_image = self.current_image.copy()
            
            if filter_name == "Grayscale":
                filtered_image = filtered_image.convert('L')
            elif filter_name == "Sharpen":
                filtered_image = filtered_image.filter(ImageFilter.SHARPEN)
            elif filter_name == "Upscale":
                # Simple upscale by 2x
                width, height = filtered_image.size
                filtered_image = filtered_image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
            elif filter_name == "CLAHE":
                # Convert to numpy array for CLAHE
                if filtered_image.mode != 'L':
                    filtered_image = filtered_image.convert('L')
                img_array = np.array(filtered_image)
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
                
                filtered_image = Image.fromarray(img_array)
            elif filter_name == "Invert":
                if filtered_image.mode == 'RGBA':
                    # Keep alpha channel
                    r, g, b, a = filtered_image.split()
                    rgb = Image.merge('RGB', (r, g, b))
                    inverted = ImageOps.invert(rgb)
                    r, g, b = inverted.split()
                    filtered_image = Image.merge('RGBA', (r, g, b, a))
                else:
                    filtered_image = ImageOps.invert(filtered_image)
            elif filter_name == "Edge Enhance":
                filtered_image = filtered_image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_name == "Find Edges":
                filtered_image = filtered_image.filter(ImageFilter.FIND_EDGES)
            elif filter_name == "Contour":
                filtered_image = filtered_image.filter(ImageFilter.CONTOUR)
            elif filter_name == "Emboss":
                filtered_image = filtered_image.filter(ImageFilter.EMBOSS)
            elif filter_name == "Enhance Contrast":
                enhancer = ImageEnhance.Contrast(filtered_image)
                filtered_image = enhancer.enhance(2.0)
            
            self.current_filtered_image = filtered_image
            
            # Update processed viewer
            qimage = self.pil_to_qimage(filtered_image)
            self.processed_viewer.set_image(QPixmap.fromImage(qimage))
            
            # Update comparison view
            qimage_orig = self.pil_to_qimage(self.current_image)
            self.comparison_viewer1.set_image(QPixmap.fromImage(qimage_orig))
            self.comparison_viewer2.set_image(QPixmap.fromImage(qimage))
            
            self.log_message(f"Applied filter: {filter_name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Filter Error", f"Could not apply filter: {str(e)}")
            logger.error(f"Filter application error: {e}")
    
    def show_ocr_dialog(self):
        """Show OCR text detection dialog"""
        if self.current_filtered_image is None:
            QMessageBox.warning(self, "No Image", "Please load and process an image first")
            return
        
        dialog = TextDetectionDialog(self)
        dialog.set_image(self.current_filtered_image)
        dialog.exec_()
    
    def show_crypto_dialog(self):
        """Show cryptographic analysis dialog"""
        dialog = CryptoAnalysisDialog(self)
        
        # Pre-fill with selected text if available
        if self.text_preview.toPlainText():
            dialog.input_text.setPlainText(self.text_preview.toPlainText())
        
        dialog.exec_()
    
    def zoom_in(self):
        """Zoom in on current image viewer"""
        current_tab = self.image_tabs.currentWidget()
        if current_tab == self.original_tab:
            self.original_viewer.scale(1.2, 1.2)
        elif current_tab == self.processed_tab:
            self.processed_viewer.scale(1.2, 1.2)
        elif current_tab == self.comparison_tab:
            self.comparison_viewer1.scale(1.2, 1.2)
            self.comparison_viewer2.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out on current image viewer"""
        current_tab = self.image_tabs.currentWidget()
        if current_tab == self.original_tab:
            self.original_viewer.scale(0.8, 0.8)
        elif current_tab == self.processed_tab:
            self.processed_viewer.scale(0.8, 0.8)
        elif current_tab == self.comparison_tab:
            self.comparison_viewer1.scale(0.8, 0.8)
            self.comparison_viewer2.scale(0.8, 0.8)
    
    def reset_zoom(self):
        """Reset zoom on current image viewer"""
        current_tab = self.image_tabs.currentWidget()
        if current_tab == self.original_tab:
            self.original_viewer.reset_zoom()
        elif current_tab == self.processed_tab:
            self.processed_viewer.reset_zoom()
        elif current_tab == self.comparison_tab:
            self.comparison_viewer1.reset_zoom()
            self.comparison_viewer2.reset_zoom()
    
    def export_report(self):
        """Export analysis report"""
        if not self.analysis_result:
            QMessageBox.warning(self, "No Data", "No analysis results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Report",
            f"treasure_hunt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.analysis_result, f, indent=2, default=str)
                
                QMessageBox.information(self, "Export Successful", f"Report exported to: {file_path}")
                self.log_message(f"Exported report: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export report: {str(e)}")
                logger.error(f"Export error: {e}")
    
    def log_message(self, message: str):
        """Add message to logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.append(f"[{timestamp}] {message}")
    
    def clear_results(self):
        """Clear all results"""
        self.results_tree.clear()
        self.text_candidates_list.clear()
        self.text_preview.clear()
        self.ai_widget.suggestion_text.clear()
        self.ai_widget.next_steps_list.clear()
        self.analysis_result = None
    
    def pil_to_qimage(self, pil_image: Image.Image) -> QImage:
        """Convert PIL Image to QImage - FIXED VERSION"""
        try:
            if pil_image.mode == "RGB":
                # Convert RGB to BGR for QImage
                r, g, b = pil_image.split()
                pil_image = Image.merge("RGB", (b, g, r))
                buffer = pil_image.tobytes("raw", "RGB")
                qimage = QImage(buffer, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
            elif pil_image.mode == "RGBA":
                # Convert RGBA to BGRA for QImage
                r, g, b, a = pil_image.split()
                pil_image = Image.merge("RGBA", (b, g, r, a))
                buffer = pil_image.tobytes("raw", "RGBA")
                qimage = QImage(buffer, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
            elif pil_image.mode == "L":
                # Grayscale
                buffer = pil_image.tobytes("raw", "L")
                qimage = QImage(buffer, pil_image.size[0], pil_image.size[1], QImage.Format_Grayscale8)
            else:
                # Convert to RGB as fallback
                pil_image = pil_image.convert("RGB")
                r, g, b = pil_image.split()
                pil_image = Image.merge("RGB", (b, g, r))
                buffer = pil_image.tobytes("raw", "RGB")
                qimage = QImage(buffer, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
            
            return qimage.copy()  # Return copy to avoid memory issues
            
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            # Return a blank image on error
            return QImage(100, 100, QImage.Format_RGB888)
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop any running threads
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
        
        # Confirm close if analysis is running
        if self.analysis_thread and self.analysis_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Analysis is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main application entry point"""
    # Set Wayland environment variables
    os.environ['QT_QPA_PLATFORM'] = 'wayland;xcb'  # Fallback to X11 if Wayland fails
    os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'  # Reduce Wayland debug output
    
    # Check for dependencies
    try:
        import pytesseract
        import pyzbar
        logger.info("All dependencies found")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install pytesseract pyzbar-py opencv-python pillow")
        return 1
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application attributes for Wayland compatibility
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = TreasureHuntGUI()
    window.show()
    
    # Handle Wayland-specific initialization
    try:
        # Check if we're running on Wayland
        if os.environ.get('WAYLAND_DISPLAY'):
            logger.info("Running on Wayland")
        else:
            logger.info("Running on X11 or other platform")
    except Exception as e:
        logger.warning(f"Platform detection failed: {e}")
    
    # Start event loop
    try:
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()