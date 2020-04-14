import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    #찾은 얼굴이 없으면 None으로 리턴
    if faces is():
        return None
    #얼굴들이 있으면
    for(x,y,w,h) in faces:
        #해당 얼굴 크기만큼 cropped_face에 잘라 넣기
        #근데... 얼굴이 2개 이상 감지되면??
        #가장 마지막의 얼굴만 남을 듯
        cropped_face = img[y:y+h, x:x+w]
    #cropped_face 리턴
    cropped_face = cv2.resize(cropped_face, (200, 200))
    return cropped_face


data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []

count = 0
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if face_extractor(image) is not None:
        count += 1
        img = cv2.resize(face_extractor(image), (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name_path = 'facess/user' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, img)
    else:
        print("Face not found")


data_path = 'facess/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        continue
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
if len(Labels) == 0:
    print("There is no data to train.")
    exit()
Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")

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