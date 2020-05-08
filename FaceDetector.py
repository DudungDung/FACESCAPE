# face detection with mtcnn on a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
print("Detector Initialize")


def detect_faces(pixels):
    return detector.detect_faces(pixels)
