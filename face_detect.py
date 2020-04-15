import cv2

def face_detect(image_file):

    cascade_file = "C:\\Users\\sec\\Desktop\\haarcascade_frontalface_alt.xml"

    image = cv2.imread(image_file)
    print(image)

    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_file)

    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(150, 150))

    if len(face_list) > 0:
        print(face_list)
        color = (0, 0, 255)
        for face in face_list:
            x, y, w, h = face
            #print(x, y, w, h);
            face_img = image[y:y+h, x:x+w];
            #face_img = image[(x*7)//5:y+(h*4)//5, (x*7)//5:x+(w*4)//5];
            face_img = cv2.resize(face_img, (w//20, h//20));
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA);
            image[y:y+h, x:x+h] = face_img
           # cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness = 8)

            adr = input("저장주소 :: ")
            cv2.imwrite(adr, image)

    else:
        print("no face")

image_file = input("이미지 파일 :: ")

face_detect(image_file)