from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from urllib import request
from io import BytesIO
from PIL import Image

vgg_model = VGG16(include_top = "True", weights = "imagenet")
vgg_model = Model(inputs = vgg_model.input, outputs = vgg_model.layers[-2].output)

def load_image(path):
    data = []
    label = []
    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            label.append(file_name)
            img_name = os.path.join(path,file_name)
            img = image.load_img(img_name, target_size = (224,224))
            img_data = image.img_to_array(img)
            data.append(img_data)
    data = np.asarray(data)
    deep_feature = vgg_model.predict(data)
    print("Loading Image Completed!")
    return deep_feature, label

data, label = load_image('data')
query, anh = load_image('query')
print(data.shape)


def cosine_similarity(x,y):
    return np.dot(x,y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))

def L2_norm(x,y):
    return np.sqrt(np.sum((x-y) ** 2))

def L1_norm(x,y):
    return np.sum(np.abs(x-y))

def mAP(data, classes):
    AP = []
    count_Overall_AP = 0
    if classes == 0:
        classes = 'human'
        for i in range(len(data),0,-1):
            TP = 0
            if data[i-1][:5] == "human":
                count_Overall_AP += 1
            for j in range(len(data[:i])):
                if data[j][:5] == 'human':
                    TP += 1
            Precision = TP / len(data)
            AP.append(TP)

    elif classes == 1:
        classes = 'cat'
        for i in range(len(data),0,-1):
            TP = 0
            if data[i-1][:3] == "cat":
                count_Overall_AP += 1
            for j in range(len(data[:i])):
                if data[j][:3] == 'cat':
                    TP += 1
            Precision = TP / len(data)
            AP.append(TP)

    elif classes == 2:
        classes = 'dog'
        for i in range(len(data),0,-1):
            TP = 0
            if data[i-1][:3] == "dog":
                count_Overall_AP += 1
            for j in range(len(data[:i])):
                if data[j][:3] == 'dog':
                    TP += 1
            Precision = TP / len(data)
            AP.append(TP)

    elif classes == 3:
        classes = 'panda'
        for i in range(len(data),0,-1):
            TP = 0
            if data[i-1][:5] == "panda":
                count_Overall_AP += 1
            for j in range(len(data[:i])):
                if data[j][:5] == 'panda':
                    TP += 1
            Precision = TP / len(data)
            AP.append(TP)
    else:
        classes = 'tiger'
        for i in range(len(data),0,-1):
            TP = 0
            if data[i-1][:5] == "tiger":
                count_Overall_AP += 1
            for j in range(len(data[:i])):
                if data[j][:5] == 'tiger':
                    TP += 1
            Precision = TP / len(data)
            AP.append(TP)

    AP = np.asarray(AP)
    AP = AP/count_Overall_AP
    MAP = np.mean(AP)
    return MAP


def Metrics(q,data, k, classes):

    L2 = np.asarray([L2_norm(x,q) for x in data])
    # metric = sorted(L2)[:k]

    L2 = L2.argsort()[:k]

    print('L2 Metrics:')
    predict = []
    for i in L2:
        predict.append(label[i])
    MAP = mAP(predict, classes)
    print(MAP)

    L1 = [L1_norm(x,q) for x in data]
    # metric = sorted(L1)[:k]

    L1 = np.argsort(L1)[:k]
    print('L1 Metrics:')
    predict = []
    for i in L1:
        predict.append(label[i])
    MAP = mAP(predict, classes)
    print(MAP)

    cosine = [cosine_similarity(x,q) for x in data]
    # metric = sorted(cosine,reverse=True)[:k]

    cosine = np.argsort(cosine)[::-1][:k]
    print('Cosine Similarity Metrics:')
    predict = []
    for i in cosine:
        predict.append(label[i])
    MAP = mAP(predict, classes)
    print(MAP)

    dot = [np.dot(x,q) for x in data]
    # metric = sorted(dot,reverse=True)[:k]

    dot = np.argsort(dot)[::-1][:k]
    print('Dot Product Metrics:')
    predict = []
    for i in dot:
        predict.append(label[i])
    MAP = mAP(predict, classes)
    print(MAP)

    print('----'*30 + '\n' )


# Ảnh class Human
print("Thong tin truy van: anh1.jpg")
print(query[0])
Metrics(query[0],data, k = 20, classes = 0)

print("Thong tin truy van: anh2.jpg")
print(query[1])
Metrics(query[1],data, k = 20, classes = 0)

# Ảnh class cat
print("Thong tin truy van: anh3.jpg")
print(query[2])
Metrics(query[2],data, k = 23, classes = 1)

# Ảnh class dog
print("Thong tin truy van: anh4.jpg")
print(query[3])
Metrics(query[3],data, k = 22, classes = 2)

# Ảnh class panda
print("Thong tin truy van: anh5.jpg")
print(query[4])
Metrics(query[4],data, k = 22, classes = 3)

# Ảnh class tiger
print("Thong tin truy van: anh6.jpg")
print(query[5])
Metrics(query[5],data, k = 18, classes = 4)
