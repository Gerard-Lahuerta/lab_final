import os
import tensorflow
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



def obtain_results(file_path:str) -> list[list[int]]:
    sudoku = []
    file = open(file_path, "r")

    for i,line in enumerate(file):
        if i > 1:
            row = line.split(" ")
            sudoku.append(row)
    
    return sudoku




if __name__ == '__main__':
    path = "v2_train/"
    train_images = []
    train_labels = []
    for i in os.listdir(path):
        if ".dat" in i:
            train_labels.append(obtain_results(path+i))
        else:
            train_images.append(cv2.imread(path+i))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    eval_result = model.evaluate(test_images, test_labels)
    print("Test Accuracy: {:.2f}%".format(eval_result[1] * 100))

    prediction = model.predict(new_sudoku_image)