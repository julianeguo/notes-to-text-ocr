# core.py
# OCR + PDF extraction logic
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pypdf import PdfReader
from PIL import Image
import torch
import io
