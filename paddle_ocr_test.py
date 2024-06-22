import cv2
from paddleocr import PaddleOCR

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the image
img_path = 'captured_image.png'

# Read the image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# Perform OCR on the image
result = ocr.ocr(img_path, cls=True)

# Extract and print the text
for line in result:
    plain_text = " ".join([item[1][0] for item in line])
    print(plain_text)
