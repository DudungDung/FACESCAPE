import cv2
from moviepy.editor import *

import face_clustering as fc
import face_detect as fd
import pickle
import time

import os
import stat
from os import listdir
from os.path import isfile, join

# 파일 확장자 알아내기
"""
 파일 확장자는 가장 끝부분에 .***의 형식으로 되어 있다.
 따라서 파일 경로를 거꾸로 뒤집은 뒤 .이 나올 때 까지 반복문을 돌리고
 나오는 순간 반복문을 탈출하여 거꾸로 저장해나간 문자열을 다시 뒤집으면
 그 파일의 확장자가 된다.
"""

# 경로 입력 받은 후 파일 유무 확인
movieExtensionList = ['.mkv', '.avi', '.mp4', '.mpg', '.flv', '.wmv']


def Codec_To_String(cc):
    cc = int(cc)
    codec_str = ""
    for i in range(4):
        codec_str = codec_str + chr((cc >> 8 * i) & 0xFF)
    return codec_str


def Find_Extension(filePath):
    ext = ''
    for c in filePath[::-1]:
        ext = ext + c
        if c == '.':
            break
    ext = ext[::-1]
    return ext


def Check_File(filePath):
    try:
        if os.path.isfile(filePath):
            extension = Find_Extension(filePath)
            # 파일이 없을 경우에는 괜찮지만 동영상 파일이 아닐 경우에는 제대로 작동하지 않을 수 있음.
            if extension not in movieExtensionList:
                return "동영상 파일이 아닙니다."
            else:
                return None
        else:
            return "파일이 없습니다."

    except FileNotFoundError:
        return "파일이 없습니다."


def Check_Directory(dirPath):
    try:
        if os.path.isdir(dirPath):
            return None
        else:
            return "잘못된 경로입니다"
    except FileNotFoundError:
        return "잘못된 경로입니다."


def Edit_Movie(filePath, fileName, extension, name):
    # model = pickle.loads(open(f"data/model/{name}.model", "rb").read())
    movieData = cv2.VideoCapture(filePath)

    # 출력 결과 파일을 data폴더에 temp.* 파일로 폴더에 저장
    videoCodec = int(movieData.get(cv2.CAP_PROP_FOURCC))
    width = movieData.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = movieData.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = movieData.get(cv2.CAP_PROP_FPS)
    tempFileName = "data/Temp" + extension
    output = cv2.VideoWriter(tempFileName, videoCodec, fps, (int(width), int(height)))

    number = 1

    maxFrame = movieData.get(cv2.CAP_PROP_FRAME_COUNT)

    dirName = "data/IMG/Video/"
    '''
    start = time.time()
    if os.path.exists(dirName):
        files = [f for f in listdir(dirName) if isfile(join(dirName, f))]
        for i, file in enumerate(files):
            os.chmod(dirName + files[i], stat.S_IWUSR)
            os.remove(dirName + files[i])
        os.rmdir(dirName)
    end = time.time()
    print(f"Remove All Files :{end - start}s")
    '''
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    start = time.time()
    while movieData.isOpened():
        ret, frame = movieData.read()
        if frame is None:
            break
        print(f"Save Frame in Video {number} / {maxFrame}")
        cv2.imwrite(dirName + "IMG" + f"{number:04}" + ".jpg", frame)
        number += 1
        # if number > 1000:
        #   break
    end = time.time()
    print(f"Time to save video: {end - start: .2f}s")

    fc.sk_clustering(dirName)
    return

    maxFrame = movieData.get(cv2.CAP_PROP_FRAME_COUNT)
    movieData = cv2.VideoCapture(filePath)
    # 영상 읽기
    while movieData.isOpened():
        ret, frame = movieData.read()
        if frame is None:
            break

        output.write(fd.face_detect(frame, model))

        # 다음 프레임으로 진행
        number = number + 1
        print(f'Progress: {number} / {maxFrame}')
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
    videoFile.write_videofile(fileName + extension)
    os.remove(tempFileName)
