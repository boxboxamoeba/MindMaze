# treasure_hunt_gui.py
import sys
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QLabel, QTextEdit, QListWidget, QListWidgetItem,
                             QPushButton, QTabWidget, QGroupBox, QScrollArea, QFrame,
                             QProgressBar, QFileDialog, QMessageBox, QCheckBox, QComboBox,
                             QLineEdit, QSpinBox, QDoubleSpinBox, QToolBar, QStatusBar,
                             QAction, QTreeWidget, QTreeWidgetItem, QHeaderView, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRectF
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon, QPen, QBrush
import PyQt5.QtGui as QtGui

# Wayland Compatibility
os.environ['QT_QPA_PLATFORM'] = 'wayland'

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# Import your existing backend functions
from treasure_hunt_pipeline import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingThread(QThread):
    """Thread for running analysis to keep UI responsive"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress_signal.emit(10, "Starting analysis...")
            result = analyze(self.file_path)
            self.progress_signal.emit(100, "Analysis complete")
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))

class ImageViewer(QGraphicsView):
    """Custom graphics view for image display with zoom and pan"""
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        
        self.zoom_factor = 1.0
        self.current_pixmap = None
        
    def set_image(self, pixmap):
        """Set the image to display"""
        self.scene.clear()
        self.current_pixmap = pixmap
        if pixmap:
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.fitInView(pixmap_item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0
    
    def wheelEvent(self, event):
        """Handle zoom with mouse wheel"""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_in_factor
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.zoom_factor *= zoom_out_factor
            self.scale(zoom_out_factor, zoom_out_factor)
    
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
    
    def set_image(self, image):
        """Set image for OCR analysis"""
        self.current_image = image
    
    def retry_ocr(self):
        """Retry OCR with current settings"""
        if self.current_image:
            try:
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
                
                # Get confidence (simplified)
                data = pytesseract.image_to_data(self.current_image, lang=lang, output_type=pytesseract.Output.DICT)
                if data['conf']:
                    avg_conf = sum(conf for conf in data['conf'] if conf > 0) / len([conf for conf in data['conf'] if conf > 0])
                    self.confidence_label.setText(f"{avg_conf:.1f}%")
                else:
                    self.confidence_label.setText("N/A")
                    
            except Exception as e:
                QMessageBox.warning(self, "OCR Error", f"Failed to perform OCR: {str(e)}")

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
            return
        
        analysis_type = self.freq_type.currentText()
        
        if analysis_type == "Single Character":
            # Count single characters
            from collections import Counter
            counter = Counter(c for c in text if c.isalpha())
            total = sum(counter.values())
            
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
                actual = (counter[char] / total * 100) if total > 0 else 0
                expected = english_freq.get(char, 0)
                diff = actual - expected
                result += f"{char}: {actual:5.2f}% (expected {expected:4.1f}%, diff: {diff:6.2f}%)\n"
                
        self.freq_results.setPlainText(result)
    
    def decrypt_cipher(self):
        """Decrypt using selected cipher"""
        text = self.input_text.toPlainText()
        cipher_type = self.cipher_type.currentText()
        key = self.key_input.text()
        
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
                result += f"Shift {shift:2}: {decrypted}\n"
                
        elif cipher_type == "Atbash":
            result = "Atbash Decryption:\n\n"
            result += atbash(text.encode()).decode() if text else ""
            
        elif cipher_type == "Vigenère":
            if not key:
                QMessageBox.warning(self, "Missing Key", "Please enter a Vigenère key")
                return
            result = f"Vigenère Decryption (key: {key}):\n\n"
            result += vigenere_decrypt(text, key)
        
        self.decrypt_results.setPlainText(result)
    
    def brute_force(self):
        """Perform brute force attack"""
        text = self.input_text.toPlainText()
        attack_type = self.attack_type.currentText()
        
        result = f"Brute Force Results ({attack_type}):\n\n"
        
        if attack_type == "Caesar (all shifts)":
            # Same as Caesar decryption above
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
    
    def update_suggestions(self, analysis_result):
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
        self.current_image_variants = {}
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Treasure Hunt Analysis Suite")
        self.setGeometry(100, 50, 1600, 1000)
        
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
        self.browse_btn.setIcon(QIcon.fromTheme("document-open"))
        file_controls.addWidget(self.browse_btn)
        
        self.batch_mode = QCheckBox("Batch Mode")
        file_controls.addWidget(self.batch_mode)
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
            "Edge Enhance", "Find Edges", "Contour", "Emboss"
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
        open_icon = QIcon.fromTheme("document-open")
        open_action = QAction(open_icon, "Open File", self)
        open_action.triggered.connect(self.browse_files)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        analyze_icon = QIcon.fromTheme("system-run")
        analyze_action = QAction(analyze_icon, "Analyze", self)
        analyze_action.triggered.connect(self.analyze_file)
        toolbar.addAction(analyze_action)
        
        toolbar.addSeparator()
        
        export_icon = QIcon.fromTheme("document-save")
        export_action = QAction(export_icon, "Export", self)
        export_action.triggered.connect(self.export_report)
        toolbar.addAction(export_action)
    
    def setup_connections(self):
        """Setup signal connections"""
        # File browser
        self.browse_btn.clicked.connect(self.browse_files)
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
        
        # Drag and drop
        self.setAcceptDrops(True)
    
    def browse_files(self):
        """Open file dialog to select files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Files to Analyze", 
            "", 
            "All Files (*);;Images (*.png *.jpg *.jpeg *.bmp *.tiff);;Archives (*.zip *.rar *.7z);;Documents (*.pdf *.docx *.xlsx)"
        )
        
        if files:
            self.file_list.clear()
            for file in files:
                self.file_list.addItem(file)
            
            # Select first file
            if self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)
                self.on_file_selected(self.file_list.item(0))
    
    def browse_folder(self):
        """Open folder dialog to select folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Analyze")
        
        if folder:
            self.file_list.clear()
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.file_list.addItem(file_path)
            
            # Select first file
            if self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)
                self.on_file_selected(self.file_list.item(0))
    
    def on_file_selected(self, item):
        """Handle file selection from list"""
        if item:
            self.current_file = item.text()
            self.analyze_btn.setEnabled(True)
            self.apply_filter_btn.setEnabled(False)
            
            # Display file info
            file_info = f"File: {os.path.basename(self.current_file)}\n"
            file_info += f"Path: {self.current_file}\n"
            file_info += f"Size: {os.path.getsize(self.current_file)} bytes\n"
            
            self.file_info.setPlainText(file_info)
            
            # Try to display image if it's an image file
            try:
                img = Image.open(self.current_file)
                img.verify()  # Verify it's a valid image
                img = Image.open(self.current_file)  # Reopen after verify
                
                # Convert to QPixmap and display
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                data = img.tobytes("raw", "RGB")
                qim = QImage(data, img.width, img.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qim)
                
                self.original_viewer.set_image(pixmap)
                self.processed_viewer.set_image(pixmap)
                self.apply_filter_btn.setEnabled(True)
                
                # Store original image for processing
                self.current_image = img
                
            except Exception as e:
                # Not an image or error loading
                self.original_viewer.set_image(None)
                self.processed_viewer.set_image(None)
                self.apply_filter_btn.setEnabled(False)
                logger.warning(f"Could not display file as image: {e}")
    
    def analyze_file(self):
        """Start analysis of current file"""
        if not self.current_file:
            return
        
        # Clear previous results
        self.results_tree.clear()
        self.text_candidates_list.clear()
        self.text_preview.clear()
        self.logs_text.clear()
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.analyze_btn.setEnabled(False)
        
        # Start analysis thread
        self.analysis_thread = ProcessingThread(self.current_file)
        self.analysis_thread.progress_signal.connect(self.on_analysis_progress)
        self.analysis_thread.finished_signal.connect(self.on_analysis_complete)
        self.analysis_thread.error_signal.connect(self.on_analysis_error)
        self.analysis_thread.start()
        
        # Log start
        self.log_message(f"Starting analysis of: {os.path.basename(self.current_file)}")
    
    def on_analysis_progress(self, progress, message):
        """Handle analysis progress updates"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)
        self.log_message(f"Progress: {progress}% - {message}")
    
    def on_analysis_complete(self, result):
        """Handle analysis completion"""
        self.analysis_result = result
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # Display results
        self.display_analysis_results(result)
        
        # Update AI suggestions
        self.ai_widget.update_suggestions(result)
        
        # Log completion
        self.log_message("Analysis complete")
        self.status_bar.showMessage("Analysis complete")
    
    def on_analysis_error(self, error_message):
        """Handle analysis errors"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Analysis Error", f"Error during analysis: {error_message}")
        self.log_message(f"ERROR: {error_message}")
        self.status_bar.showMessage("Analysis failed")
    
    def display_analysis_results(self, result):
        """Display analysis results in the tree widget"""
        # File info
        file_item = QTreeWidgetItem(["File Info", f"Type: {result.get('file_type', 'Unknown')}", ""])
        self.results_tree.addTopLevelItem(file_item)
        
        # Text candidates
        if result.get('text_candidates'):
            text_parent = QTreeWidgetItem(["Text Candidates", f"{len(result['text_candidates'])} found", ""])
            self.results_tree.addTopLevelItem(text_parent)
            
            for candidate in result['text_candidates']:
                method = candidate.get('method', 'Unknown')
                text = candidate.get('text', '')[:50] + "..." if len(candidate.get('text', '')) > 50 else candidate.get('text', '')
                score = candidate.get('score', 0)
                
                text_item = QTreeWidgetItem([method, text, f"{score:.2f}"])
                text_parent.addChild(text_item)
        
        # QR/Barcodes
        if result.get('qr_barcode_outputs'):
            qr_parent = QTreeWidgetItem(["QR/Barcodes", f"{len(result['qr_barcode_outputs'])} found", ""])
            self.results_tree.addTopLevelItem(qr_parent)
            
            for qr in result['qr_barcode_outputs']:
                qr_item = QTreeWidgetItem(["QR/Barcode", qr.get('data', '')[:100], ""])
                qr_parent.addChild(qr_item)
        
        # Steganography
        if result.get('stego_candidates'):
            stego_parent = QTreeWidgetItem(["Steganography", f"{len(result['stego_candidates'])} candidates", ""])
            self.results_tree.addTopLevelItem(stego_parent)
            
            for stego in result['stego_candidates']:
                stego_item = QTreeWidgetItem(["Stego", stego.get('method', 'Unknown'), f"{stego.get('confidence', 0):.2f}"])
                stego_parent.addChild(stego_item)
        
        # Hidden data
        if result.get('hidden_unicode'):
            unicode_item = QTreeWidgetItem(["Hidden Unicode", f"{len(result['hidden_unicode'])} characters", ""])
            self.results_tree.addTopLevelItem(unicode_item)
        
        # Update text candidates list
        if result.get('text_candidates'):
            self.text_candidates_list.clear()
            for i, candidate in enumerate(result['text_candidates']):
                method = candidate.get('method', 'Unknown')
                score = candidate.get('score', 0)
                self.text_candidates_list.addItem(f"{i+1}. {method} (score: {score:.2f})")
    
    def on_text_candidate_selected(self, item):
        """Handle text candidate selection"""
        if not self.analysis_result or not self.analysis_result.get('text_candidates'):
            return
        
        index = self.text_candidates_list.currentRow()
        if 0 <= index < len(self.analysis_result['text_candidates']):
            candidate = self.analysis_result['text_candidates'][index]
            self.text_preview.setPlainText(candidate.get('text', ''))
    
    def apply_image_filter(self):
        """Apply selected image filter"""
        if not hasattr(self, 'current_image') or not self.current_image:
            return
        
        filter_name = self.filter_combo.currentText()
        
        try:
            if filter_name == "Original":
                filtered_img = self.current_image
            elif filter_name == "Grayscale":
                filtered_img = self.current_image.convert('L').convert('RGB')
            elif filter_name == "Sharpen":
                filtered_img = self.current_image.filter(ImageFilter.SHARPEN)
            elif filter_name == "Upscale":
                # Simple upscale
                width, height = self.current_image.size
                filtered_img = self.current_image.resize((width*2, height*2), Image.Resampling.LANCZOS)
            elif filter_name == "CLAHE":
                # Convert to OpenCV for CLAHE
                img_cv = np.array(self.current_image.convert('RGB'))
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                lab_planes = list(cv2.split(lab))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                filtered_img = Image.fromarray(img_cv)
            elif filter_name == "Invert":
                filtered_img = ImageOps.invert(self.current_image.convert('RGB'))
            elif filter_name == "Edge Enhance":
                filtered_img = self.current_image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_name == "Find Edges":
                filtered_img = self.current_image.filter(ImageFilter.FIND_EDGES)
            elif filter_name == "Contour":
                filtered_img = self.current_image.filter(ImageFilter.CONTOUR)
            elif filter_name == "Emboss":
                filtered_img = self.current_image.filter(ImageFilter.EMBOSS)
            
            # Convert to QPixmap and display
            if filtered_img.mode != 'RGB':
                filtered_img = filtered_img.convert('RGB')
            
            data = filtered_img.tobytes("raw", "RGB")
            qim = QImage(data, filtered_img.width, filtered_img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)
            
            self.processed_viewer.set_image(pixmap)
            
            # Store filtered image
            self.current_filtered_image = filtered_img
            
            self.log_message(f"Applied filter: {filter_name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Filter Error", f"Failed to apply filter: {str(e)}")
            logger.error(f"Filter error: {e}")
    
    def show_ocr_dialog(self):
        """Show OCR text detection dialog"""
        if not hasattr(self, 'current_image') or not self.current_image:
            QMessageBox.information(self, "No Image", "Please load an image file first.")
            return
        
        dialog = TextDetectionDialog(self)
        dialog.set_image(self.current_image)
        
        # If we have a filtered image, use that
        if hasattr(self, 'current_filtered_image'):
            dialog.set_image(self.current_filtered_image)
        
        dialog.exec_()
    
    def show_crypto_dialog(self):
        """Show cryptographic analysis dialog"""
        dialog = CryptoAnalysisDialog(self)
        
        # Pre-populate with text from analysis if available
        if self.analysis_result and self.analysis_result.get('text_candidates'):
            best_candidate = self.analysis_result['text_candidates'][0]
            dialog.input_text.setPlainText(best_candidate.get('text', ''))
        
        dialog.exec_()
    
    def zoom_in(self):
        """Zoom in on current image"""
        current_viewer = self.get_current_viewer()
        if current_viewer:
            current_viewer.scale(1.25, 1.25)
            current_viewer.zoom_factor *= 1.25
    
    def zoom_out(self):
        """Zoom out on current image"""
        current_viewer = self.get_current_viewer()
        if current_viewer:
            current_viewer.scale(0.8, 0.8)
            current_viewer.zoom_factor *= 0.8
    
    def reset_zoom(self):
        """Reset zoom on current image"""
        current_viewer = self.get_current_viewer()
        if current_viewer:
            current_viewer.reset_zoom()
    
    def get_current_viewer(self):
        """Get the current active image viewer based on tab"""
        current_tab = self.image_tabs.currentIndex()
        
        if current_tab == 0:  # Original
            return self.original_viewer
        elif current_tab == 1:  # Processed
            return self.processed_viewer
        elif current_tab == 2:  # Comparison
            # For comparison, return the first viewer
            return self.comparison_viewer1
        
        return None
    
    def log_message(self, message):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.append(f"[{timestamp}] {message}")
    
    def export_report(self):
        """Export analysis report"""
        if not self.analysis_result:
            QMessageBox.information(self, "No Data", "No analysis data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Report", 
            f"treasure_hunt_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 
            "JSON Files (*.json);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                # Create report data
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'file_analyzed': self.current_file,
                    'analysis_results': self.analysis_result,
                    'logs': self.logs_text.toPlainText()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                self.log_message(f"Report exported to: {file_path}")
                QMessageBox.information(self, "Export Successful", f"Report exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export report: {str(e)}")
    
    def dragEnterEvent(self, event):
        """Handle drag enter event for file drops"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop event for file drops"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self.file_list.clear()
                self.file_list.addItem(file_path)
                self.file_list.setCurrentRow(0)
                self.on_file_selected(self.file_list.item(0))

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Decrpter")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Treasure Hunt Team")
    
    # Create and show main window
    window = TreasureHuntGUI()
    window.show()
    
    # Start application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()