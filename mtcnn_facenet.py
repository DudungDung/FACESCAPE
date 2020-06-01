from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray, load
from numpy import expand_dims
from numpy import load
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from random import choice


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(directory):
    x, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue

        faces = load_faces(path)

        labels = [subdir for _ in range(len(faces))]

        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('int32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predice(samples)
    return yhat[0]


# dataset -> trainX, trainY, testX, testY
# embedding -> newTrainX, trainY, newTestX, testY
trainX, trainY = load_dataset('data/model/')
testX, testY = load_dataset('data/test/')

model = load_model('model/facenet_keras.h5')

newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)

newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model, face_pixels)
    newTestX.append(embedding)
newTestX = asarray(newTestX)

txfa = trainY

in_encoder = Normalizer(norm='10')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = txfa[selection]
random_face_emb = testX[selection]
random_face_class = testY[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('예상 : %s (%.3f)' % (predict_names[0], class_probability))
print('추측 : %s' % random_face_name[0])

pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
