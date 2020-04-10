import cv2
from moviepy.editor import *

import GuiManager

# 파일 확장자 알아내기
"""
 파일 확장자는 가장 끝부분에 .***의 형식으로 되어 있다.
 따라서 파일 경로를 거꾸로 뒤집은 뒤 .이 나올 때 까지 반복문을 돌리고
 나오는 순간 반복문을 탈출하여 거꾸로 저장해나간 문자열을 다시 뒤집으면
 그 파일의 확장자가 된다.
"""


def codec_to_string(cc):
    cc = int(cc)
    codec_str = ""
    for i in range(4):
        codec_str = codec_str + chr((cc >> 8 * i) & 0xFF)
    return codec_str


def FindExtension(str):
    ext = ''
    for c in str[::-1]:
        ext = ext + c
        if c == '.':
            break
    ext = ext[::-1]
    return ext


# 경로 입력 받은 후 파일 유무 확인
movieExtensionList = ['.mkv', '.avi', '.mp4', '.mpg', '.flv', '.wmv']

while True:
    filePath = GuiManager.LoadMovieFile()
    print(filePath)
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
        print('''1. 프레임 새기기 2. 프레임 가리기 3. 프레임 삭제(X)''')
        selType = int(input("숫자를 입력하세요."))
        if (selType < 1) or (selType > 3):
            print('''1 ~ 2까지의 숫자를 입력하세요.''')
        else:
            break
    except ValueError:
        print('''숫자를 입력해주세요.''')

# 출력 결과 파일을 data폴더에 temp.* 파일로 폴더에 저장
videoCodec = int(movieData.get(cv2.CAP_PROP_FOURCC))
width = movieData.get(cv2.CAP_PROP_FRAME_WIDTH)
height = movieData.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = movieData.get(cv2.CAP_PROP_FPS)
fileName = 'output' + extension
tempFileName = "data/Temp"+extension
output = cv2.VideoWriter(tempFileName, videoCodec, fps, (int(width), int(height)))

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

    # 3번일 경우 아무런 처리도 해주지 않음
    elif selType == 3:
        if 30 > number or number > 80:
            output.write(frame)

    # 다음 프레임으로 진행
    number = number + 1
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# OpenCV를 통해 영상처리가 끝난 파일들을 release해줌
movieData.release()
output.release()
cv2.destroyAllWindows()

# 영상에 소리를 입히기 위해서 moviePy를 이용하여 파일을 저장. 임시로 만들어둔 data/temp.*파일은 삭제
audioClip = AudioFileClip(filePath)
videoClip = VideoFileClip(tempFileName)
videoFile = videoClip.set_audio(audioClip)
videoFile.write_videofile(fileName)
os.remove(tempFileName)
