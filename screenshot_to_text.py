import pytesseract
from PIL import Image
import pyautogui

# Point to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 1️⃣ Capture screenshot
screenshot = pyautogui.screenshot()
screenshot.save("capture.png")

# 2️⃣ Extract text from screenshot
extracted_text = pytesseract.image_to_string(Image.open("capture.png"))

# 3️⃣ Save extracted text to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("✅ Text extracted and saved to output.txt")


