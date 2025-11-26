# main.py
# where the functions get ran
from core import load_model, image_to_text, line_segmentation
from PIL import Image
import numpy as np
import cv2

def main():
    print("main ran")
    processor, model = load_model()

    image_path = 'samples/hello_world_test_2.png' ###* sample image
    image = cv2.imread(image_path)

    line_images = line_segmentation(image)
    
    for i in range(len(line_images)):
        # convert bgr to rgb
        line_rgb = cv2.cvtColor(line_images[i], cv2.COLOR_BGR2RGB)
        # convert rgb to pil for the next function
        line_pil = Image.fromarray(line_rgb)
        
        generated_text = image_to_text(line_pil, processor, model)

if __name__ == "__main__":
    main()
