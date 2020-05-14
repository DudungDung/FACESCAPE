import os
import time

import dlib
import numpy as np
from os import listdir
from os.path import isfile, join

import face_detect as fd

from sklearn.cluster import DBSCAN
import face_recognition

import cv2


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


def sk_clustering():
    videoPath = "data/IMG/Video/"

    onlyfiles = [f for f in listdir(videoPath) if isfile(join(videoPath, f))]

    data = []
    for i, img in enumerate(onlyfiles):
        imgPath = videoPath + onlyfiles[i]
        img = dlib.load_rgb_image(imgPath)
        '''
        box = face_recognition.face_locations(img, model="hog")
        encoding = face_recognition.face_encodings(img, box)
        for b, e in zip(box, encoding):
            boxes.append(b)
            encodings.append(e)
        '''
        if i != 0:
            end = time.time()
            print(f"Find Face {i}/{len(onlyfiles)}: {end - start: .2f}s")
        start = time.time()
        faces_dnn, h, w = fd.find_faces_dnn(img)
        for j in range(0, faces_dnn.shape[2]):
            confidence = faces_dnn[0, 0, j, 2]
            if confidence > 0.5:
                dnn_box = faces_dnn[0, 0, j, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = dnn_box.astype("int")
                box = [[sy, ex, ey, sx]]
                encoding = face_recognition.face_encodings(img, box)
                d = {"frameNum": i, "box": box[0], "encoding": encoding}
                data.append(d)

    encodings = [d["encoding"] for d in data]

    encoding_new = []
    for e in encodings:
        encoding_new.append(np.array(e).flatten())

    clt = DBSCAN(metric="euclidean")
    clt.fit(encoding_new)

    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])

    dirName = "data/IMG/Learning/Temp"
    print("Saving faces in largest cluster to output folder...")
    start = time.time()
    for label_id in label_ids:
        print(f"Clustering Face {i + 1} / {len(label_ids)}")
        path = dirName + str(label_id) + '/'
        if not os.path.isdir(path):
            os.makedirs(path)

        face_indexes = np.where(clt.labels_ == label_id)[0]

        for i in face_indexes:
            frame_id = data[i]["frameNum"]
            box = data[i]["box"]

            imgPath = videoPath + onlyfiles[frame_id]
            img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
            faceImage = img[box[0]: box[2], box[3]: box[1]]
            filePath = os.path.join(path, "face_" + str(i) + ".jpg")
            cv2.imwrite(filePath, faceImage)
    end = time.time()
    print(f"Cluster All Image: {end - start: .2f}")


def Img_clustering():
    videoPath = "data/IMG/Video/"
    dirName = "data/IMG/Learning/Temp"

    recognition_path = "data/dlib_face_recognition_resnet_model_v1.dat"
    predictor_path = "data/shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    faceRec = dlib.face_recognition_model_v1(recognition_path)

    onlyfiles = [f for f in listdir(videoPath) if isfile(join(videoPath, f))]

    descriptors = []
    shapes = []

    for i, img in enumerate(onlyfiles):

        imgPath = videoPath + onlyfiles[i]
        img = dlib.load_rgb_image(imgPath)

        if i != 0:
            end = time.time()
            print(f"Find Face {i}/{len(onlyfiles)}: {end - start: .2f}s")
        start = time.time()
        faces, h, w = fd.find_faces_dnn(img)
        for j in range(0, faces.shape[2]):
            confidence = faces[0, 0, j, 2]
            if confidence > 0.5:
                dnn_box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = dnn_box.astype("int")
                box = dlib.rectangle(left=sx, top=sy, right=ex, bottom=ey)
                print(box)
                shape = predictor(img, box)
                face_descriptor = faceRec.compute_face_descriptor(img, shape)
                descriptors.append(face_descriptor)
                shapes.append((i, shape))

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
    start = time.time()
    for i, label in enumerate(labels):
        print(f"Clustering Face {i+1} / {len(labels)}")
        index, shape = shapes[i]
        imgPath = videoPath + onlyfiles[index]
        img = dlib.load_rgb_image(imgPath)
        path = dirName + str(label) + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        file_path = os.path.join(path, "face_" + str(i))
        # The size and padding arguments are optional with default size=150x150 and padding=0.25
        dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
    end = time.time()
    print(f"Cluster All Image: {end - start: .2f}")
