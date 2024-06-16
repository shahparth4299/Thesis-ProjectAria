# easy_ocr_read.py
import os
import certifi
import easyocr

def read_text_from_image(image_path):
    # Set the SSL certificate file path
    os.environ['SSL_CERT_FILE'] = certifi.where()

    # Initialize the easyocr Reader
    reader = easyocr.Reader(['en'])  # specify the language

    # Read text from the image
    result = reader.readtext(image_path)

    # Store all recognized text in a list
    recognized_texts = [text for (bbox, text, prob) in result]

    # Join all recognized text into a single string
    all_text = " ".join(recognized_texts)

    return all_text
