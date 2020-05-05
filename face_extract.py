import cv2
import numpy as np

import os
from os import listdir
from os.path import isfile, join


def face_extractor(name):  ## 호출시 이름 입력
    data_path = 'data/IMG/' + name + '/'

    if not os.path.exists(data_path):
        print("사진 폴더가 존재하지 않습니다.")
        return

    face_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

    count = 0
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    if onlyfiles.__len__() <= 0:
        print("사진이 존재하지 않습니다.")
        return
    Training_Data, Labels = [], []
    image_path = data_path + onlyfiles[0]  # 폴더 내의 사진의 1번은 특정이라 생각함
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 한번 머신러닝
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(0)
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        Labelf = np.asarray(Labels, dtype=np.int32)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.asarray(Training_Data), np.asarray(Labelf))

        dirName = 'data/Learning/' + name + '/'
        try:
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Create Directory: " + dirName)

        except OSError:
            print("Error: Creating directory: " + dirName)

        i = 1
        for (x, y, w, h) in faces:
            print('Getting face ' + i)
            cropped_face = image[y:y + h, x:x + w]
            img = cv2.resize(cropped_face, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = model.predict(img)
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                if confidence > 73:  # 유사도가 73퍼 이상이면 넣어줌
                    print('Get ' + name + ' face')
                    img = cv2.resize(cropped_face, (200, 200))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    count += 1
                    file_name_path = dirName + 'Learning_IMG' + str(count) + '.jpg'
                    cv2.imwrite(file_name_path, img)
                    Training_Data.append(np.asarray(img, dtype=np.uint8))
                    Labels.append(count)
                    Labelf = np.asarray(Labels, dtype=np.int32)
                    model = cv2.face.LBPHFaceRecognizer_create()
                    model.train(np.asarray(Training_Data), np.asarray(Labelf))
            else:
                print("Face not found")
