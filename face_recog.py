import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.cluster import DBSCAN
import os
import sys

def face_detector(img, size = -1.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


def face_extractor():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
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


face_extractor()

data_path = 'facess/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
Training_Data, Labels = [], []
for i, image in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")
"""
print(model)
def FindExtension(str):
    ext = ''
    for c in str[::-1]:
        ext = ext + c
        if c == '.':
            break
    ext = ext[::-1]
    return ext

movieExtensionList = ['.mkv', '.avi', '.mp4', '.mpg', '.flv', '.wmv', '.asf', '.asx', '.ogm', '.ogv', '.mov']
while True:
    filePath = input("파일 경로를 입력하세요.")
    try:
        f = open(filePath, 'r')
        extension = FindExtension(filePath)
        # 파일이 없을 경우에는 괜찮지만 동영상 파일이 아닐 경우에는 제대로 작동하지 않을 수 있음.
        if extension in movieExtensionList:
            movieData = cv2.VideoCapture(filePath)
            break
        else:
            print("동영상 파일이 아닙니다.")
    except FileNotFoundError:
        print("파일이 없습니다.")

selType = 0
#저장
codec = int(movieData.get(cv2.CAP_PROP_FOURCC))
width = movieData.get(cv2.CAP_PROP_FRAME_WIDTH)
height = movieData.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = movieData.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('output' + extension, codec, fps, (int(width), int(height)))

# 테스트용 putText 속성들
number = 1
loc = (30, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
black = (0, 0, 0)
thickness = 2


while movieData.isOpened():
    ret, frame = movieData.read()
    # 얼굴 검출 시도
    image, face = face_detector(frame)
    if frame is None:
        break
    try:
        #검출된 사진을 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #위에서 학습한 모델로 예측시도
        result = model.predict(face)
        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
        # 유사도 화면에 표시
        display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        #75 보다 크면 동일 인물로 간주해 UnLocked!
        if confidence > 75:
           # cv2.putText(frame, "Frame 1", loc, font, fontScale, black, thickness)
           # output.write(frame)
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
           # cv2.imshow('Face Cropper', image)
            output.write(image)
        else:
            #75 이하면 타인.. Locked!!!
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            #cv2.imshow('Face Cropper', image)
            output.write(image)
    except:
    #얼굴 검출 안됨
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #cv2.imshow('Face Cropper', image)
        pass
    if cv2.waitKey(1) == 13 and (0xFF == ord('q')):
        break
movieData.release()
output.release()
cv2.destroyAllWindows()
"""