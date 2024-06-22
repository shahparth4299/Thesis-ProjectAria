import os
import certifi
from paddleocr import PaddleOCR

# Set the SSL certificate file path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def read_text_from_image(image_path):
    # Perform OCR on the image
    result = ocr.ocr(image_path, cls=True)

    # Extract and concatenate recognized text
    recognized_texts = []
    for line in result:
        plain_text = " ".join([item[1][0] for item in line])
        recognized_texts.append(plain_text)

    # Join all recognized text into a single string
    all_text = " ".join(recognized_texts)

    return all_text

# Test
if __name__ == "__main__":
    img_path = 'captured_image.png'
    recognized_text = read_text_from_image(img_path)
    print("Recognized Text:")
    print(recognized_text)
