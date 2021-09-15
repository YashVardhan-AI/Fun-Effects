# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf



def get_mesh(model_path='models/model.tflite'):

    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Set model input
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Preprocess the image before sending to the network.



    return interpreter, input_details, output_details


def get_square_box(box):
    # Get a square box out of the given box, by expanding it
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'
    
    return [left_x, top_y, right_x, bottom_y]


def move_box(box, offset):
    # Move the box to direction specified by vector offset
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def detect_marks(img, face):

    offset_y = int(abs((face[3] - face[1]) * 0.1))
    box_moved = move_box(face, [0, offset_y])
    facebox = get_square_box(box_moved)

    h, w = img.shape[:2]
    if facebox[0] < 0:
        facebox[0] = 0
    if facebox[1] < 0:
        facebox[1] = 0
    if facebox[2] > w:
        facebox[2] = w
    if facebox[3] > h:
        facebox[3] = h

    try:
        face_img = img[facebox[1]: facebox[3],
                       facebox[0]: facebox[2]]

        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)


        image = tf.image.convert_image_dtype(face_img, tf.uint8)
        image = np.expand_dims(image, axis=0)
        interpreter, input_details, output_details = get_mesh()
        # The actual detection.
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()

        # Save the results.
        mesh = interpreter.get_tensor(output_details[0]["index"])[
            0]
    
    # Convert predictions to landmarks.
        marks = np.array(mesh).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))

        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        marks = marks.astype(np.uint)
        
        return marks
    except Exception as e:
        pass
# define  a function to draw masks


def draw_marks(image, marks, color=(0, 255, 0)):
    for mark in marks:
        img = cv2.circle(image, (mark[0], mark[1]), 1, color, -1, cv2.LINE_AA)

    return img


def line(img, marks):
    img = cv2.drawContours(img, [marks], 0, (255, 255, 255), 1)
    return img


def linemain(img, marks):
    for index, item in enumerate(marks):
        if index == len(marks) - 1:
            break
        img = cv2.line(img, item, marks[index + 1], [255, 255, 255], 1)
    return img
