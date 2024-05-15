
# IMPORTS
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import k_means 

import tensorflow as tf
from tensorflow.keras import layers, models, datasets


# FUNCTIONS
def detect_large_square(image_path:str) -> np.array:
    '''
    Detect the biggest square a image and returning the cut of the square

    Input:
        -> image_path (str): path of the image
    
    Return:
        -> np.array of the cutted square image
    '''

    # upload image
    image = cv2.imread(image_path)
    
    # gray scale and detection of countours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_square = None
    max_area = 0

    # search of the biggest square
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                large_square = approx

    # crop the square
    if large_square is not None:
        x_coords = [point[0][0] for point in large_square]
        y_coords = [point[0][1] for point in large_square]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    else: 
        print("No large square found.")
        return None

def find_intersections(lines):
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        rho, theta = line[0]
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
                intersections.append((int(x), int(y)))
            except np.linalg.LinAlgError:
                continue
    return intersections

def crop_cells(image, intersections):
    cells = []
    model = k_means(X = intersections, n_clusters = 100)[0]
    model = sorted(model, key=lambda x: (x[0], x[1]))
    for x1, y1 in model:
        for x2, y2 in model:
            if x1==x2 or y1==y2: continue
            if 0.8 < (y2-y1)/(x2-x1) < 1.2 and 50 < y2-y1 < 102:  
                cell_image = image[int(y1):int(y2), int(x1):int(x2)]
                # if 0 < cell_image.size < 102:  
                cells.append(cell_image)
    return cells


def detect_and_crop_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        print("No lines were detected.")
        return []
    intersections = find_intersections(lines)
    cells = crop_cells(image, intersections)
    return cells


def get_CNN(x_train: list[np.ndarray], y:list[list[int]]):
    y_train = np.concatenate(y)
    x_train = np.array([reshape_image(i) for i in x_train])

    model = models.Sequential([
        layers.Reshape((160, 160, 1), input_shape=(160, 160)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size = 9)

    return model


def reshape_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    m, n = gray.shape

    new_image = np.zeros((160,160))

    row_offset = (160 - m) // 2
    col_offset = (160 - n) // 2

    for i in range(m):
        for j in range(n):
            new_image[row_offset + i, col_offset + j] = gray[i, j]
    
    return new_image

def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        matrix = [list(map(int, line.split())) for line in lines[2:]]
        
    return matrix


if __name__ == '__main__':
    img = detect_large_square('v2_train/image1082.jpg')
    if img is None:
        exit()

    small_imgs = detect_and_crop_cells(img)
    if len(small_imgs) != 81:
        print("Detected cells count:", len(small_imgs))
        exit()

    sudoku = read_matrix_from_file("v2_train/image1082.dat")

    model = get_CNN(small_imgs, sudoku)
    res = np.zeros((9,9))

    for i,img in enumerate(small_imgs):
        img = reshape_image(img)
        # exit()
        prediction = model.predict(x = np.array(img).reshape(1,160,160))
        print(i//9, i-i//9)
        res[i//9, i-(i//9)*9] = int(prediction)

    print(res)


