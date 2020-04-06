import cv2

# 파일 확장자 알아내기
"""
 파일 확장자는 가장 끝부분에 .***의 형식으로 되어 있다.
 따라서 파일 경로를 거꾸로 뒤집은 뒤 .이 나올 때 까지 반복문을 돌리고
 나오는 순간 반복문을 탈출하여 거꾸로 저장해나간 문자열을 다시 뒤집으면
 그 파일의 확장자가 된다.
"""


def FindExtension(str):
    ext = ''
    for c in str[::-1]:
        ext = ext + c
        if c == '.':
            break
    ext = ext[::-1]
    return ext


# 경로 입력 받은 후 파일 유무 확인
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
while True:
    try:
        print('''1. 프레임 새기기 2. 프레임 가리기 3. 프레임 삭제''')
        selType = int(input("숫자를 입력하세요."))
        if (selType < 1) or (selType > 3):
            print('''1 ~ 3까지의 숫자를 입력하세요.''')
        else:
            break
    except ValueError:
        print('''숫자를 입력해주세요.''')

# 출력 결과 파일 output.* 파일로 폴더에 저장
codec = int(movieData.get(cv2.CAP_PROP_FOURCC))
width = movieData.get(cv2.CAP_PROP_FRAME_WIDTH)
height = movieData.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = movieData.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('output' + extension, codec, fps, (int(width), int(height)))

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
