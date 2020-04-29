from urllib.request import urlopen
import requests as req
from bs4 import BeautifulSoup
import os


def Crawling_Image(name, maxAmount):
    # 이미지 url https://www.google.com/search?q=검색내용&tbm=isch
    # 구글은 기본적으로 20개를 불러오는 방식을 이용함
    # 이 때 start 인자를 이용하면 시작지점을 정할 수 있어 20개단위로 여러번 작동시켜 원하는만큼 받아오도록 함
    googleUrl = "https://www.google.com/search?"

    startNum = 0
    currentImageAmount = 1

    # 기본적으로 요청량만큼 받지만 이미지 다운로드에 실패하여 startNum이 너무 늘어날 경우 그냥 끝냄
    while currentImageAmount <= maxAmount or startNum > maxAmount * 3:
        params = {
            "q": name,
            "tbm": "isch",
            "start": startNum
        }
        htmlData = req.get(googleUrl, params)

        # 데이터 불러오는데에 성공했을 경우에 사용
        if htmlData.status_code == 200:
            soup = BeautifulSoup(htmlData.text, "html.parser")
            imgDatas = soup.find_all("img", {'src': True})

            dirName = "data/IMG/" + name + '/'

            try:
                if not os.path.exists(dirName):
                    os.makedirs(dirName)
                    print("Create Directory: " + dirName)
            except OSError:
                print("Error: Creating directory: " + dirName)

            for i in enumerate(imgDatas):
                # 구글 이미지 소스 파일 중에는 구글로고가 포함되어있는데 이는 urlopen으로 열리지 않는 이미지이다
                # 따라서 ValueError로 예외처리 해줘서 문제가 안생기도록 한다
                try:
                    img = urlopen(i[1].attrs['src']).read()
                    filename = dirName + name + str(currentImageAmount) + '.jpg'
                    with open(filename, 'wb') as f:
                        f.write(img)
                        print(i[1].attrs['src'])
                        print("Img Save Success: " + str(currentImageAmount))
                        currentImageAmount += 1
                except ValueError:
                    continue

            startNum += 20


Crawling_Image('박효신', 100)
