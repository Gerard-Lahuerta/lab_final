import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def detect_large_square(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return None, None
    
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
        x_coords = [point[0][0] for point in large_square]
        y_coords = [point[0][1] for point in large_square]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image, max_area
    else: 
        print("No large square found.")
        return None, None

def find_intersections(lines, image_shape):
    height, width = image_shape[:2]
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        for rho, theta in line:
            if theta < np.pi / 4 or theta > 3 * np.pi / 4:
                vertical_lines.append((rho, theta))
            else:
                horizontal_lines.append((rho, theta))
    intersections = []
    for h_rho, h_theta in horizontal_lines:
        for v_rho, v_theta in vertical_lines:
            a1 = np.cos(h_theta)
            b1 = np.sin(h_theta)
            a2 = np.cos(v_theta)
            b2 = np.sin(v_theta)
            matrix = np.array([[a1, b1], [a2, b2]])
            rho = np.array([h_rho, v_rho])
            try:
                x, y = np.linalg.solve(matrix, rho)
                if 0 <= x < width and 0 <= y < height:
                    intersections.append((int(x), int(y)))
            except np.linalg.LinAlgError:
                continue
    print(len(intersections))
    return intersections

def crop_cells(image, intersections):
    cells = []
    
    expected_grid_size = 20 

    intersections_matrix = [intersections[i:i + expected_grid_size] for i in range(0, len(intersections), expected_grid_size)]

    for i in range(expected_grid_size - 1):
        for j in range(expected_grid_size - 1):
            x1, y1 = intersections_matrix[i][j]
            x2, y2 = intersections_matrix[i+1][j+1]
            if x1==x2 or y1==y2: continue
            if x1 > x2: x1, x2 = x2, x1 
            if y1 > y2: y1, y2 = y2, y1 
            if 0.8 < (y2-y1)/(x2-x1) < 1.2:  
                cell_image = image[y1:y2, x1:x2]
                if cell_image.size > 0:  
                    cells.append(cell_image)
    return cells


def detect_and_crop_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        print("No lines were detected.")
        return []
    intersections = find_intersections(lines, image.shape)
    cells = crop_cells(image, intersections)
    return cells


def get_CNN():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = models.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    return model


if __name__ == '__main__':
    img, max_area = detect_large_square('v2_train/image1087.jpg')
    if img is None or max_area is None:
        exit()
    
    cv2.imshow('Cropped Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    small_imgs = detect_and_crop_cells(img)
    print("Detected cells count:", len(small_imgs))
    if len(small_imgs) == 0:
        exit()

    for i in small_imgs:
        if i.size > 0:
            cv2.imshow('Cropped Image', i)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Empty or invalid cell image detected.")

    #######

    model = get_CNN()
    sudoku = np.zeros(9)

    for i,img in enumerate(small_imgs):
        img = img.reshape(1, 28, 28) / 255.0

        predictions = model.predict(img)
        if max(predictions) < 0.75:
            sudoku[i//9][i-i//9] = -1
        else:
            number = np.argmax(predictions)
            sudoku[i//9][i-i//9] = number

    print(sudoku)


