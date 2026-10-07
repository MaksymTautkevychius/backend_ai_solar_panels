import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_satellite_for_sam(image_path):
    # 1. Завантаження
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # 2. Mean Shift Filtering (Спрощення кольорів для супутника)
    # Це прибирає текстуру даху, роблячи його однорідним
    shifted = cv2.pyrMeanShiftFiltering(img_bgr, sp=15, sr=40)

    # 3. Покращення контрасту (CLAHE)
    lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 4. Морфологічне очищення (ВИПРАВЛЕННЯ ПОМИЛКИ)
    # Створюємо ядро і ЯВНО вказуємо тип uint8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Використовуємо морфологію для "склеювання" масиву даху
    final_bgr = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # 5. Підготовка для SAM та відображення (RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    
    # Вивід результату
    plt.figure(figsize=(10, 10))
    plt.imshow(final_rgb)
    plt.title("Cleaned Image for SAM")
    plt.axis('off')
    
    # Якщо ви в WSL2 і немає вікна - збережіть у файл:
    plt.savefig("preprocessed_output.png")
    print("Preprocessed image saved as preprocessed_output.png")
    
    plt.show()

    return final_rgb





def detect_roof_edges(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny — best general edge detector
    edges_canny = cv2.Canny(blurred, threshold1=30, threshold2=90)
    
    # Structured Edge Detection alternative:
    # Works much better on satellite textures
    # edges = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    
    # Close small gaps in roof outline
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernel)
    plt.figure(figsize=(10, 10))
    plt.imshow(edges_closed)
    plt.title("Cleaned Image for SAM")
    plt.axis('off')
    
    # Якщо ви в WSL2 і немає вікна - збережіть у файл:
    plt.savefig("preprocessed_output.png")
    print("Preprocessed image saved as preprocessed_output.png")
    
    plt.show()
    
    return edges_closed









if __name__ == "__main__":
    processed_img = detect_roof_edges("./uploads/62.png")

