import cv2
import numpy as np

# 경로 입력 받은 후 파일 유무 확인
while True:
    filePath = input("파일 경로를 입력하세요.")
    try:
        f = open(filePath, 'r')
        movieData = cv2.VideoCapture(filePath)
        break
    except FileNotFoundError:
        print("파일이 없습니다.")


selType = 0
while True:
    try:
        print('''1. 프레임 새기기 2. 프레임 가리기 3. 프레임 삭제''')
        selType = int(input("숫자를 입력하세요."))
        if (selType < 1) or (selType > 3):
            print('''1 ~ 3까지의 숫자를 입력하세요''')
        else:
            break
    except ValueError:
        print('''숫자를 입력해주세요''')

# 결과 출력
codec = cv2.VideoWriter_fourcc(*'DIVX')
width = movieData.get(cv2.CAP_PROP_FRAME_WIDTH)
height = movieData.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = movieData.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('output.avi', codec, fps, (int(width), int(height)))

# 프레임가리는 사각형
loadedImg = cv2.imread("data/BlackImage.jpg")
blackRec = cv2.resize(loadedImg, (int(width), (int(height))))

# 테스트용 putText 속성들
number = 1
loc = (30, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
black = (0, 0, 0)
thickness = 2

# 영상 읽기
while movieData.isOpened():
    ret, frame = movieData.read()

    if frame is None:
        break

    if selType == 1:
        # 영상 프레임 매기기
        cv2.putText(frame, "Frame " + repr(number), loc, font, fontScale, black, thickness)
        output.write(frame)

    elif selType == 2:
        # 프레임 가리기(임시조건)
        if 30 < number < 80:
            output.write(blackRec)
        else:
            output.write(frame)

    elif selType == 3:
        if 30 > number or number > 80:
            output.write(frame)

    # 다음 프레임으로 진행
    number = number + 1
    # 3번일 경우 아무런 처리도 해주지 않음
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

movieData.release()
output.release()
cv2.destroyAllWindows()
