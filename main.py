import cv2

image_file = "C:\\Users\\sec\\Desktop\\good2.jpg"

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
        face_img = image[x:y+h, x:x+w];
        face_img = cv2.resize(face_img, (w//20, h//20));
        face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA);
        image[y:y+h, x:x+h] = face_img
        #cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=8)

        cv2.imwrite("C:\\Users\\sec\\Desktop\\good-good2.jpg", image)

else:
    print("no face")