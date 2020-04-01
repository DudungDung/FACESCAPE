import cv2


# 경로 입력 받은 후 파일 유무 확인
while True:
    filePath = input("파일 경로를 입력하세요.")
    movieData = cv2.VideoCapture(filePath)
    if movieData.isOpened() == False:
        print("파일이 없습니다.")
    else:
        break

# 결과
codec = cv2.VideoWriter_fourcc(*'DIVX')
width = movieData.get(cv2.CAP_PROP_FRAME_WIDTH)
height = movieData.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = movieData.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('output.avi', codec, fps, (int(width), int(height)))

# 테스트용 putText 속성들
number = 1;
loc = (30,50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
black = (0,0,0)
thickness = 2

# 영상 읽기
while movieData.isOpened():
    ret, frame = movieData.read();

    if frame is None:
        break

    # 영상 프레임 매기기
    cv2.putText(frame, "Frame " + repr(number), loc, font, fontScale, black, thickness)
    print("Process Frame" + repr(number))
    number = number + 1

    cv2.imshow('frame', frame)

    output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

movieData.release()
output.release()
cv2.destroyAllWindows()
