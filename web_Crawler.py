import errno
from urllib.request import urlopen
import requests as req
from bs4 import BeautifulSoup
import os


def Crawling_Image(name):
    # 이미지 url https://www.google.com/search?q=검색내용&tbm=isch
    googleUrl = "https://www.google.com/search?"

    params = {
        "q": name,
        "tbm": "isch"
    }

    html_object = req.get(googleUrl,params)

    if html_object.status_code == 200:
        bs_object = BeautifulSoup(html_object.text, "html.parser")
        img_data = bs_object.find_all("img", {'src': True}, limit=100)

        dirName = "data/IMG/" + name + '/'
        try:
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Create Directory: " + dirName)
        except OSError:
            print("Error: Creating directory: " + dirName)

        for i in enumerate(img_data[1:]):
            t = urlopen(i[1].attrs['src']).read()
            filename = dirName + name + str(i[0] + 1) + '.jpg'
            with open(filename, 'wb') as f:
                f.write(t)
                print("Img Save Success" + str(i[0] + 1))

Crawling_Image('아이유')