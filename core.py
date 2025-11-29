# core.py
# OCR + PDF extraction logic
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrOCRConfig, TrOCRForCausalLM
import requests
from pypdf import PdfReader
from PIL import Image
import numpy as np
import cv2

# loads the model and returns the processor and model 
def load_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    return processor, model

# removes the 
#!charGPT wrote this function
def preprocess_img(gray):
    # gray: single-channel grayscale image

    # 1) Inverse binary: text + lines = 255, background = 0
    ret, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bin_inv.shape
    kernel_width = max(40, w // 25) #!
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))

    # 2) Detect horizontal lines
    horizontal = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)

    # 3) Remove those lines from the *grayscale* image
    gray_no_lines = gray.copy()
    mask = (horizontal == 255)   # places where we detected lines
    gray_no_lines[mask] = 255    # paint them white

    cv2.imwrite("debug_horizontal.png", horizontal)
    cv2.imwrite("debug_bin_inv.png", bin_inv)

    return gray_no_lines



# performs line segmentation
def line_segmentation(image):
    # 1. convert image to grayscale
    if len(image.shape) == 3:
        grayed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayed_img = image  # already grayscale

    grayed_img = preprocess_img(grayed_img) #! remove lines

    # 2. binarize it
    ret, bin_img = cv2.threshold(grayed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ink_mask = (bin_img == 0) # mask to find all the pixels with ink
    ink_pixel_counts = np.sum(ink_mask, axis=1) # this works because we have either 0 or 1, and sum gives us num of 1s!

    # print(ink_pixel_counts[:20]) # testing

    # 3. HPP
    threshold = 0.1 * ink_pixel_counts.max()
    in_line = False
    start_pix = 0
    end_pix = 0
    lines = [] # tuples of start, end of lines
    for i in range(ink_pixel_counts.shape[0]):
        if in_line == False and ink_pixel_counts[i] >= threshold: # new line start
            in_line = True
            start_pix = i
        elif in_line == True and ink_pixel_counts[i] <= threshold:
            in_line = False
            end_pix = i
            if end_pix - start_pix > 10: #? condition for actually keeping the bands
                lines.append((start_pix, end_pix))

    if in_line:
        lines.append((start_pix, ink_pixel_counts.shape[0]))

    # 4. segment the image
    line_img = []
    h, w = bin_img.shape
    pad = 17 #? padding
    for (ystart, yend) in lines:
        y0 = max(ystart - pad, 0)
        y1 = min(yend + pad, ink_pixel_counts.shape[0])
        line_img.append(bin_img[y0:y1, 0:w]) #! we use the original grayscale img

    #####? test code for writing each line into an img
    for i, crop in enumerate(line_img):
        cv2.imwrite(f"line_{i}.png", crop)

    return line_img # returns bgr array of line images


# loads in the image, processor, and model; outputs text
def image_to_text(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(generated_text)
    return generated_text