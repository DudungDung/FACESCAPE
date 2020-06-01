# face detection with mtcnn on a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
import dlib

mtcnn_detector = MTCNN()
hog_detector = dlib.get_frontal_face_detector()
# cnn_detector = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")

protextPath = "data/DNN/deploy.prototxt"
caffeModelPath = "data/DNN/res10_300x300_ssd_iter_140000.caffemodel"
dnn_detector = cv2.dnn.readNetFromCaffe(protextPath, caffeModelPath)

img = cv2.imread("data/Dummy.jpg")
mtcnn_detector.detect_faces(img)
print("Detector Initialize")