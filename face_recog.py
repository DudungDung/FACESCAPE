from imutils import paths
import face_recognition
import numpy as np
import pickle
import cv2
import os


def imread_utf8(img_path, flags):
    try:
        new_imgPath = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(new_imgPath, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite_utf8(img_path, img, params=None):
    try:
        ext = os.path.splitext(img_path)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(img_path, mode='wb') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return None


def Recog_Face(name):
    dirPath = "data/IMG/" + name
    learnPath = "data/Learning/" + name
    imagePaths = list(paths.list_images(dirPath))

    knownEncoding = []
    knownNames = []

    count = 0
    for i, imgPath in enumerate(imagePaths):
        print(f"Process {i+1}/{len(imagePaths)}")

        filename = imgPath.split(os.path.sep)[-2]

        image = imread_utf8(imgPath, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")
        for box in boxes:
            print(box)
            count += 1
            y, w, h, x = box
            cropped_face = image[y:h, x:w]
            file_name_path = learnPath + "/" + 'Learning_IMG' + f'{count:04}' + '.jpg'
            imwrite_utf8(file_name_path, cropped_face)

        encodings = face_recognition.face_encodings(rgb, boxes)

        for en in encodings:
            knownEncoding.append(en)
            knownNames.append(name)

    print("Serializing encodings")
    data = {"encodings": knownEncoding, "names": knownNames}
    f = open(f"data/model/{name}.model", "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("End")
