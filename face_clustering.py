import os
import dlib
import glob
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

