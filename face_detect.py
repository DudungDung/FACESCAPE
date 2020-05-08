import cv2
import FaceDetector as fd


def face_detect(image, model):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (0, 0, 0)
    thick = 2

    # image = cv2.imread(image_file)
    # print(image)

    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_list = fd.find_faces(image)

    if len(face_list) > 0:
        print(face_list)
        color = (0, 255, 0)
        for face in face_list:
            x, y, w, h = face
            # print(x, y, w, h);
            cropped_face = image[y:y + h, x:x + w]
            face_img = cv2.resize(cropped_face, (200, 200))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            result = model.predict(face_img)
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=3)
                cv2.putText(image, str(confidence), (x, y + h), fontface, fontscale, fontcolor, thick)
            # face_img = image[(x*7)//5:y+(h*4)//5, (x*7)//5:x+(w*4)//5];
            ''' 모자이크 처리 부분
            face_img = cv2.resize(face_img, (w // 20, h // 20));
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA);
            image[y:y + h, x:x + h] = face_img
            '''
            # cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness = 8)

            # adr = input("저장주소 :: ")
            # cv2.imwrite(adr, image)

            print("find face")
            return image

    else:
        return image

# image_file = input("이미지 파일 :: ")

# face_detect(image_file)
