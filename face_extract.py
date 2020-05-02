import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def face_extractor():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    data_path = 'faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    Training_Data, Labels = [], []
    image_path = data_path + onlyfiles[0]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
        for(x,y,w,h) in faces:
            cropped_face = image[y:y+h, x:x+w]
            img = cv2.resize(cropped_face, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = model.predict(img)
            Labelf = np.asarray(Labels, dtype=np.int32)
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(np.asarray(Training_Data), np.asarray(Labelf))
            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                if confidence > 73:
                    img = cv2.resize(cropped_face, (200, 200))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    count += 1
                    file_name_path = 'facess/user' + str(count) + '.jpg'
                    cv2.imwrite(file_name_path, img)
                    Training_Data.append(np.asarray(img, dtype=np.uint8))
                    Labels.append(count)
            else:
                print("Face not found")