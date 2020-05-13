import os
import dlib
import numpy as np
from os import listdir
from os.path import isfile, join

import face_detect as fd


def clustering(name):
    crawledPath = "data/IMG/Temp/"
    dirName = "data/IMG/" + "IU" + "/"

    try:
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Create Directory: " + dirName)

    except OSError:
        print("Error: Creating directory: " + dirName)

    recognition_path = "data/dlib_face_recognition_resnet_model_v1.dat"
    predictor_path = "data/shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    faceRec = dlib.face_recognition_model_v1(recognition_path)

    descriptors = []
    imgs = []

    onlyfiles = [f for f in listdir(crawledPath) if isfile(join(crawledPath, f))]

    for i, file in enumerate(onlyfiles):
        print(f"{i+1}/{len(onlyfiles)}")
        imgPath = crawledPath + onlyfiles[i]
        img = dlib.load_rgb_image(imgPath)

        faces = fd.find_faces_hog(img)

        for k, d in enumerate(faces):
            shape = predictor(img, d)
            face_descrpitor = faceRec.compute_face_descriptor(img, shape)
            descriptors.append(face_descrpitor)
            imgs.append((img, shape))

    labels = dlib.chinese_whispers_clustering(descriptors, 0.3)
    num_classes = len(set(labels))
    print("Number of clusters: {}", format(num_classes))

    # Find biggest class
    biggest_class = None
    biggest_class_length = 0
    for i in range(0, num_classes):
        class_length = len([label for label in labels if label == i])
        if class_length > biggest_class_length:
            biggest_class_length = class_length
            biggest_class = i

    print("Biggest cluster id number: {}".format(biggest_class))
    print("Number of faces in biggest cluster: {}".format(biggest_class_length))

    # Find the indices for the biggest class
    indices = []
    for i, label in enumerate(labels):
        if label == biggest_class:
            indices.append(i)

    print("Indices of images in the biggest cluster: {}".format(str(indices)))

    # Ensure output directory exists
    if not os.path.isdir(dirName):
        os.makedirs(dirName)

    # Save the extracted faces
    print("Saving faces in largest cluster to output folder...")
    for i, index in enumerate(indices):
        img, shape = imgs[index]
        file_path = os.path.join(dirName, "face_" + str(i))
        # The size and padding arguments are optional with default size=150x150 and padding=0.25
        dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)


def Img_clustering():
    dirPath = "data/IMG/Video/"
    dirName = "data/IMG/Learning/Temp"

    recognition_path = "data/dlib_face_recognition_resnet_model_v1.dat"
    predictor_path = "data/shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    faceRec = dlib.face_recognition_model_v1(recognition_path)

    onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

    descriptors = []
    imgs = []

    for i, img in enumerate(onlyfiles):
        print(f"Find Face {i+1}/{len(onlyfiles)}")

        imgPath = dirPath + onlyfiles[i]
        img = dlib.load_rgb_image(imgPath)

        faces, h, w = fd.find_faces_dnn(img)
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                dnn_box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = dnn_box.astype("int")
                box = dlib.rectangle(left=sx, top=sy, right=ex, bottom=ey)
                print(box)
                shape = predictor(img, box)
                face_descrpitor = faceRec.compute_face_descriptor(img, shape)
                descriptors.append(face_descrpitor)
                imgs.append((img, shape))

    labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
    num_classes = len(set(labels))
    print("Number of clusters: {}", format(num_classes))

    # Find biggest class
    biggest_class = None
    biggest_class_length = 0
    for i in range(0, num_classes):
        class_length = len([label for label in labels if label == i])
        if class_length > biggest_class_length:
            biggest_class_length = class_length
            biggest_class = i

    print("Biggest cluster id number: {}".format(biggest_class))
    print("Number of faces in biggest cluster: {}".format(biggest_class_length))

    # Ensure output directory exists

    # Save the extracted faces
    print("Saving faces in largest cluster to output folder...")
    for i, index in enumerate(labels):
        print(f"Clustering Face {i} / {len(labels)}")
        img, shape = imgs[i]
        dirPath = dirName + str(index) + '/'
        if not os.path.isdir(dirPath):
            os.makedirs(dirPath)
        file_path = os.path.join(dirPath, "face_" + str(i))
        # The size and padding arguments are optional with default size=150x150 and padding=0.25
        dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)

