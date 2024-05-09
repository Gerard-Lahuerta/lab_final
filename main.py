import cv2
import numpy as np

def detect_large_square(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_square = None
    max_area = 0
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                large_square = approx
    
    if large_square is not None:
        cv2.drawContours(image, [large_square], 0, (0, 255, 0), 5)
        cv2.imshow('Large Square Detected', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No large square found.")

if __name__ == '__main__':
    detect_large_square('v2_train/image9.jpg')
