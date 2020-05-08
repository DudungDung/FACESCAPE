import cv2
import FaceDetector as detector
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


def face_detect(image, model):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (255, 255, 255)
    thick = 2

    # image = cv2.imread(image_file)
    # print(image)
    face_list = find_faces_img(image)

    if len(face_list) > 0:
        print(face_list)
        color = (0, 255, 0)
        for face in face_list:
            x, y, w, h = face['box']
            if x <= 0 or y <= 0:
                print("Pass This Frame")
                continue
            # print(x, y, w, h);
            cropped_face = image[y:y + h, x:x + w]
            face_img = cv2.resize(cropped_face, (200, 200))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            result = model.predict(face_img)
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=3)
                cv2.rectangle(image, (x, y + h), (x + w, y + 3), color, thickness=3)
                cv2.putText(image, str(confidence), (x, y + h + 2), fontface, fontscale, fontcolor, thick)
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

# draw an image with detected objects
def find_faces(filename):
    pixels = pyplot.imread(filename)
    # detect faces in the image path
    faces = detector.detect_faces(pixels)
    return faces
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


def find_faces_img(img):
    # detect faces in the image
    faces = detector.detect_faces(img)
    return faces