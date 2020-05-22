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


def imread_utf8(img_path, flags):
    try:
        new_imgPath = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(new_imgPath, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite_utf8(img_path, img, params=None):
    try:
        ext = os.path.splitext(img_path)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(img_path, mode='wb') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return None


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


def sk_clustering(selectedPath):

    onlyfiles = [f for f in listdir(selectedPath) if isfile(join(selectedPath, f))]

    data = []
    st = time.time()
    for i, img in enumerate(onlyfiles):
        imgPath = selectedPath + onlyfiles[i]
        img = imread_utf8(imgPath, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        '''
        if i != 0:
            end = time.time()
            print(f"Find Face {i}/{len(onlyfiles)}: {end - start: .2f}s")
        start = time.time()
        box = face_recognition.face_locations(rgb_img, model="hog")
        encoding = face_recognition.face_encodings(rgb_img, box)
        for b, e in zip(box, encoding):
            d = {"frameNum": i, "box": b, "encoding": e}
            data.append(d)
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
                box = [sy, ex, ey, sx]
                rgb_img = cv2.resize(rgb_img, dsize=(150, 150), interpolation=cv2.INTER_LINEAR)
                rgb_box = [[0, 149, 149, 0]]
                encoding = face_recognition.face_encodings(rgb_img, rgb_box)
                d = {"frameNum": i, "box": box, "encoding": encoding}
                data.append(d)

    ed = time.time()
    print(f"All Faces Find: {ed - st:.2f}s")
    encodings = [d["encoding"] for d in data]

    encoding_new = []
    for e in encodings:
        encoding_new.append(np.array(e).flatten())

    clt = DBSCAN(metric="euclidean")
    clt.fit(encoding_new)

    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])
    print(f"Find {num_unique_faces} Faces")

    dirName = "data/IMG/Learning/Temp"
    print("Saving faces in largest cluster to output folder...")
    start = time.time()
    for label_id in label_ids:
        path = dirName + str(label_id) + '/'
        if not os.path.isdir(path):
            os.makedirs(path)

        face_indexes = np.where(clt.labels_ == label_id)[0]
        count = 0
        for i in face_indexes:
            count += 1
            print(f"Clustering Face {i + 1} / {len(face_indexes)} in ID {label_id}")
            frame_id = data[i]["frameNum"]
            box = data[i]["box"]

            imgPath = selectedPath + onlyfiles[frame_id]
            img = imread_utf8(imgPath, cv2.IMREAD_COLOR)
            faceImage = img[box[0]: box[2], box[3]: box[1]]
            filePath = os.path.join(path, "face_" + str(i) + ".jpg")
            imwrite_utf8(filePath, faceImage)
    end = time.time()
    print(f"Cluster All Image: {end - start: .2f}")


def Img_clustering(selectedPath):
    dirName = "data/IMG/Learning/Temp"

    recognition_path = "data/dlib_face_recognition_resnet_model_v1.dat"
    predictor_path = "data/shape_predictor_5_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    faceRec = dlib.face_recognition_model_v1(recognition_path)

    onlyfiles = [f for f in listdir(selectedPath) if isfile(join(selectedPath, f))]

    descriptors = []
    shapes = []

    for i, img in enumerate(onlyfiles):

        imgPath = selectedPath + onlyfiles[i]
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
        imgPath = selectedPath + onlyfiles[index]
        img = dlib.load_rgb_image(imgPath)
        path = dirName + str(label) + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        file_path = os.path.join(path, "face_" + str(i))
        # The size and padding arguments are optional with default size=150x150 and padding=0.25
        dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
    end = time.time()
    print(f"Cluster All Image: {end - start: .2f}")
