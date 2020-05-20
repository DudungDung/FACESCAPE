import numpy as np
import cv2
import FaceDetector as models
import face_recognition
import imutils
import os

import time


def face_detect(image, model):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    fontcolor = (255, 255, 255)
    thick = 2

    # image = cv2.imread(image_file)
    # print(image)

    '''
    # CNN에서 얼굴 찾기
    face_cnn_list = find_faces_cnn(image)
    color = (255, 0, 0)
    for i, face in enumerate(face_cnn_list):
        # CNN
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=3)
        cv2.putText(image, f"CNN{int(face.confidence)}", (x, y + h), fontface, fontscale, color, thick)
    '''

    '''
    # HOG에서 얼굴 찾기
    face_hog_list = find_faces_hog(image)
    if len(face_hog_list) > 0:
        color = (0, 255, 0)
        for face in face_hog_list:
            # HOG
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=3)
            cv2.putText(image, "HOG", (x, y + h), fontface, fontscale, color, thick)
            '''
    '''
            cropped_face = image[y:y + h, x:x + w]
            face_img = cv2.resize(cropped_face, (200, 200))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            result = model.predict(face_img)
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                
            모자이크 처리 부분
            face_img = cv2.resize(face_img, (w // 20, h // 20));
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA);
            image[y:y + h, x:x + h] = face_img

            cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness = 8)

            adr = input("저장주소 :: ")
            cv2.imwrite(adr, image)
    '''
    '''
    # MTCNN에서 얼굴 찾기
    face_mtcnn_list = find_faces_mtcnn(image)
    start = time.time()
    if len(face_mtcnn_list) > 0:
        color = (255, 255, 0)
        for face in face_mtcnn_list:
            x, y, w, h = face['box']
            
            box = (y, x+w, y+h, x)
            boxes = {box}
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb, boxes)

            name = "unknown"
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(model["encodings"],
                                                         encoding, tolerance=0.6)
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for n in matchedIdxs:
                        name = model["names"][n]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

            if name is not "unknown":
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=3)

    end = time.time()
    print("Face Matches: ", format(end - start, '.2f'))
    '''

    # DNN에서 얼굴 찾기
    detections_dnn, h, w = find_faces_dnn(image)
    dnn_face_amount = 0
    start = time.time()
    for i in range(0, detections_dnn.shape[2]):
        confidence = detections_dnn[0, 0, i, 2]
        if confidence > 0.15:
            dnn_face_amount += 1
            dnn_box = detections_dnn[0, 0, i, 3:7] * np.array([w, h, w, h])
            sx, sy, ex, ey = dnn_box.astype("int")
            box = [(sy, ex, ey, sx)]

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb, box)

            name = "unknown"
            for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(model["encodings"],
                                                         encoding, tolerance=0.4)
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for n in matchedIdxs:
                        name = model["names"][n]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)

            if name is not "unknown":
                color = (0, 0, 0)
                if str(name) == "hwasa":
                    color = (255, 255, 0)
                elif str(name) == "wheein":
                    color = (255, 0, 255)
                elif str(name) == "solar":
                    color = (0, 0, 255)
                elif str(name) == "munbyeol":
                    color = (0, 255, 255)

                cv2.rectangle(image, (sx, sy), (ex, ey), color, thickness=3)
                cv2.putText(image, name, (sx, ey), fontface, fontscale, color, thick)
    end = time.time()
    print("Face Matches: ", format(end - start, '.2f'))

    '''
    for i in range(0, detections_dnn.shape[2]):
        confidence = detections_dnn[0, 0, i, 2]
        if confidence > 0.15:
            dnn_face_amount += 1
            box = detections_dnn[0, 0, i, 3:7] * np.array([w, h, w, h])
            sx, sy, ex, ey = box.astype("int")
            cv2.rectangle(image, (sx, sy), (ex, ey), color, thickness=3)
            cv2.putText(image, f"DNN {int(confidence * 100)}", (sx, ey), fontface, fontscale, color, thick)
    '''
    # print(f"Find {len(face_cnn_list)} on CNN")
    # print(f"Find {len(face_hog_list)} on HOG")
    # print(f"Find {len(face_mtcnn_list)} on MTCNN")
    print(f"Find {dnn_face_amount} on DNN confidence > 15%")

    viewed_img = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5)
    cv2.imshow("example", viewed_img)
    return image


# draw an image with detected objects
def find_faces(filepath):
    img = cv2.imread(filepath)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (200, 200)), 1.0, (200, 200), (104.0, 177.0, 123.0))

    models.dnnModel.setInput(blob)
    detections = models.dnnModel.forward()

    return detections
    # detect faces in the image path
    '''
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in faces:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots on eyes nose ..
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    '''


def find_faces_hog(img):
    start = time.time()
    faces = models.hog_detector(img, 1)
    end = time.time()
    print("HOG : ", format(end - start, '.2f'))
    return faces


def find_faces_cnn(img):
    start = time.time()
    faces = models.cnn_detector(img, 1)
    end = time.time()
    print("CNN : ", format(end - start, '.2f'))
    return faces


def find_faces_mtcnn(img):
    start = time.time()
    faces = models.mtcnn_detector.detect_faces(img)
    end = time.time()
    print("MTCNN : ", format(end - start, '.2f'))
    return faces


def find_faces_dnn(img):
    start = time.time()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    models.dnn_detector.setInput(blob)
    detections = models.dnn_detector.forward()
    end = time.time()
    print("DNN : ", format(end - start, '.2f'))
    return detections, h, w


def find_one_face_dnn(filename):
    img = imread_utf8(filename)
    count = 0
    faces, h, w = find_faces_dnn(img)
    for j in range(0, faces.shape[2]):
        confidence = faces[0, 0, j, 2]
        if confidence > 0.5:
            count += 1
            if count > 1:
                break
            box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
            sx, sy, ex, ey = box.astype("int")
            faceImg = img[sy:ey, sx:ex]
            faceImg = cv2.resize(faceImg, dsize=(150, 150), interpolation=cv2.INTER_LINEAR)

    if count == 1:
        print("Find one face")
        imwrite_utf8(filename, faceImg)
        return True
    else:
        print("face is not only one")
        os.remove(filename)
        return False


def imread_utf8(img_path, flags=cv2.IMREAD_COLOR):
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

