import os
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Activation
from tensorflow.keras.utils import to_categorical

def obtain_results(file_path: str):
    sudoku = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines[2:]:  # Comenzar a leer desde la tercera línea
            row = line.strip().split()
            if row:
                sudoku.extend([int(num) for num in row])
    return np.array(sudoku)

if __name__ == '__main__':
    path = "v2_train/"
    train_images = []
    train_labels = []
    for filename in tqdm(os.listdir(path)):
        full_path = os.path.join(path, filename)
        if filename.endswith(".dat"):
            labels = obtain_results(full_path)
            train_labels.append(labels)
        else:
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (640, 480))
            img_normalized = img_resized / 255.0
            train_images.append(img_normalized)

    train_labels = np.array(train_labels)
    train_labels = to_categorical(train_labels, num_classes=10)  # Clases del 0 al 9
    train_images = np.array(train_images).reshape(-1, 480, 640, 1)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(810),  # 810 salidas lineales
        Reshape((81, 10)),  # Redimensionar a (81, 10)
        Activation('softmax')  # Aplicar softmax a cada fila de salida (81 filas)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with tf.device('/gpu:0'):
        model.fit(train_images, train_labels, epochs=10, batch_size=1, verbose=2)


    eval_result = model.evaluate(train_images, train_labels)
    print("Test Accuracy: {:.2f}%".format(eval_result[1] * 100))

    prediction = model.predict(np.expand_dims(train_images[0], axis=0))
    print(prediction.shape)  # Debería mostrar (1, 81, 10)
