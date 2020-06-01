from imutils import paths
import imutils
import face_recognition
import numpy as np
import pickle
import cv2
import os
import time
import face_detect as fd
import tensorflow as tf
from tensorflow import keras

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


def Face_Recog():
    embeddingModelPath = ""
    embedder = cv2.dnn.readNetFromTorch(embeddingModelPath)
    dirPath = "data/training-images/"
    imgPaths = list(paths.list_images(dirPath))

    knownEmbeddings = []
    knownNames = []

    total = 0
    for (i, imgPath) in enumerate(imgPaths):
        print(f"[INFO] processing image {i+1}/ {len(imgPaths)}")
        name = imgPaths.split(os.path.sep)[-2]

        image = cv2.imread(imgPath)
        image = imutils.resize(image, width=600)

        faces, h, w = fd.find_faces_dnn(image)
        for j in range(0, faces.shape[2]):
            confidence = faces[0, 0, j, 2]
            if confidence > 0.5:
                box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
                sx, sy, ex, ey = box.astype("int")

                face = image[sy:ey, sx,ex]
                fH, fW = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("마마무.model", "wb")
    f.write(pickle.dumps(data))
    f.close()


def Recog_Face():
    dirPath = "data/training-images/"
    trainNames = os.listdir(dirPath)

    knownEncodings = []
    knownNames = []

    for tName in trainNames:
        path = dirPath + tName + "/"
        imagePaths = list(paths.list_images(path))
        count = 0
        for i, imgPath in enumerate(imagePaths):
            if i != 0:
                end = time.time()
                print(f"Find {tName} Face {i + 1}/{len(imagePaths)}: {end - start: .2f}s")
            start = time.time()
            image = imread_utf8(imgPath, cv2.IMREAD_COLOR)
            faces_dnn, h, w = fd.find_faces_dnn(image)
            for j in range(0, faces_dnn.shape[2]):
                confidence = faces_dnn[0, 0, j, 2]
                if confidence > 0.5:
                    count += 1
                    dnn_box = faces_dnn[0, 0, j, 3:7] * np.array([w, h, w, h])
                    sx, sy, ex, ey = dnn_box.astype("int")
                    box = [sy, ex, ey, sx]
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, dsize=(160, 160), interpolation=cv2.INTER_LINEAR)
                    rgb_box = [[0, 149, 149, 0]]
                    encoding = face_recognition.face_encodings(rgb, rgb_box)
                    for e in encoding:
                        knownEncodings.append(e)
                        knownNames.append(tName)
    '''
    dirPath = "data/IMG/Human"
    learnPath = "data/Learning/Human"
    imagePaths = list(paths.list_images(dirPath))

    knownEncoding = []
    knownNames = []

    for i, imgPath in enumerate(imagePaths):
        print(f"Process {i + 1}/{len(imagePaths)}")

        image = imread_utf8(imgPath, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        for box in boxes:
            print(box)
            count += 1
            y, w, h, x = box
            cropped_face = image[y:h, x:w]
            file_name_path = learnPath + "/" + 'Learning_IMG' + f'{count:04}' + '.jpg'
            imwrite_utf8(file_name_path, cropped_face)

        encodings = face_recognition.face_encodings(rgb, boxes)

        for en in encodings:
            knownEncoding.append(en)
            knownNames.append("Human")
    '''

    with tf.compat.v1.Session() as sess:
        batch_join = tf.Variable(tf.concat(knownEncodings), name='batch_join')
       # encoding = tf.Variabel(tf.)
       # phase_train
        graph = tf.compat.v1.global_variables_initializer()
        sess.run(graph)
        saver = tf.compat.v1.train.Saver()

        sess.run(batch_join)
        saver.save(sess, 'data/model/model')

    print("Serializing encodings")
   # data = {"encodings": knownEncodings, "names": knownNames}
   # f = open(f"data/model/test.", "w", encoding ='utf-8')
   # f.write(pickle.dumps(data))
   # f.close()
    print("End")
