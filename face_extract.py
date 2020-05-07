import cv2
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import time


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


def face_extractor(name):  ## 호출시 이름 입력
    global model
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
    model = cv2.face.LBPHFaceRecognizer_create()

    dirName = 'data/Learning/' + name + '/'
    try:
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Create Directory: " + dirName)

    except OSError:
        print("Error: Creating directory: " + dirName)

    skipped_Number = []
    train_confidence = 75
    max_confidence = 85
    end_Amount = 0

    while count < 1000:
        for i, files in enumerate(onlyfiles):
            if i in skipped_Number:
                continue
            image_path = data_path + onlyfiles[i]
            image = imread_utf8(image_path, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            dirName = 'data/Learning/' + name + '/'
            try:
                if not os.path.exists(dirName):
                    os.makedirs(dirName)
                    print("Create Directory: " + dirName)

            except OSError:
                print("Error: Creating directory: " + dirName)

            for (x, y, w, h) in faces:
                print('Getting face from ' + image_path)
                cropped_face = image[y:y + h, x:x + w]
                img = cv2.resize(cropped_face, (200, 200))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if count == 0:
                    img = cv2.resize(cropped_face, (200, 200))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    count += 1
                    if count % 200 == 0:  # 200개단위로 증가했을 때 확률을 좁혀줌
                        max_confidence += count // 200
                    skipped_Number.insert(i, i)
                    print('Get ' + name + ' face' + count.__str__())
                    file_name_path = dirName + 'Learning_IMG' + f'{count:04}' + '.jpg'
                    imwrite_utf8(file_name_path, img)
                    Training_Data.append(np.asarray(img, dtype=np.uint8))
                    Labels.append(count)
                    Labelf = np.asarray(Labels, dtype=np.int32)
                    model.train(np.asarray(Training_Data), np.asarray(Labelf))
                    continue
                result = model.predict(img)
                if result[1] < 500:
                    confidence = int(100 * (1 - (result[1]) / 300))
                    print('Confidence :' + confidence.__str__())
                    if confidence >= train_confidence:  # 학습 시작 전에는 무조건 입력, 그 후 확률을 만족했을 경우만 넣어줌
                        img = cv2.resize(cropped_face, (200, 200))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        count += 1
                        if count % 200 == 0: # 200개단위로 증가했을 때 확률을 좁혀줌
                            max_confidence += count//200
                        skipped_Number.insert(i, i)
                        print('Get ' + name + ' face' + count.__str__())
                        file_name_path = dirName + 'Learning_IMG' + f'{count:04}' + '.jpg'
                        imwrite_utf8(file_name_path, img)
                        Training_Data.append(np.asarray(img, dtype=np.uint8))
                        Labels.append(count)
                        Labelf = np.asarray(Labels, dtype=np.int32)
                        model.train(np.asarray(Training_Data), np.asarray(Labelf))

                        if train_confidence < max_confidence:
                            train_confidence += 0.2
                            if train_confidence >= max_confidence:
                                train_confidence = int(max_confidence)

                else:
                    print("Face not found")

        if end_Amount == count:
            print("Lower confidence condition")
            train_confidence -= 1
            max_confidence -= 1
        end_Amount = count

    modelDir = 'data/model/'
    try:
        if not os.path.exists(modelDir):
            os.makedirs(modelDir)
            print("Create Directory: " + modelDir)

    except OSError:
        print("Error: Creating directory: " + modelDir)

    try:
        if os.path.exists(modelDir + name + ".model"):
            os.remove(modelDir + name + ".model:")
            print("Remove already existing file: " + name + ".model")
    except OSError:
        print("Error: Cant remove file: " + name + ".model")

    model.save(modelDir + "trainer.model")
    os.rename(modelDir + "trainer.model", modelDir + name + ".model")
    print("Save Model File: " + modelDir + name + ".model")
    print("Learning " + str(count) + "Pictures")