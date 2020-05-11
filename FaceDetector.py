# face detection with mtcnn on a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2

detector = MTCNN()

protextPath = "data/DNN/deploy.prototxt.txt"
caffeModelPath = "data/DNN/res10_300x300_ssd_iter_140000.caffemodel"
dnnModel = cv2.dnn.readNetFromCaffe(protextPath, caffeModelPath)
print("Detector Initialize")
