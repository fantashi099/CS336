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

def Metrics(q,data,count):

    fname = "Ket qua anh" + str(count) + ".txt"

    f = open('./output/' + fname, 'a')

    L2 = np.asarray([L2_norm(x,q) for x in data])
    metric = sorted(L2)[:5]
    f.write('L2 Metrics:')
    f.write(str(metric))
    f.write('\n')

    L2 = L2.argsort()[:5]

    print('L2 Metrics:')
    for i in L2:
        f.write(label[i])
        f.write('\t')
        print(label[i], end='; ')
    f.write('\n')

    print('\n')

    L1 = [L1_norm(x,q) for x in data]
    metric = sorted(L1)[:5]
    f.write('L1 Metrics:')
    f.write(str(metric))
    f.write('\n')

    L1 = np.argsort(L1)[:5]
    print('L1 Metrics:')
    for i in L1:
        f.write(label[i])
        f.write('\t')
        print(label[i], end='; ')
    f.write('\n')

    print('\n')

    cosine = [cosine_similarity(x,q) for x in data]
    metric = sorted(cosine,reverse=True)[:5]
    f.write('Cosine Similarity Metrics:')
    f.write(str(metric))
    f.write('\n')

    cosine = np.argsort(cosine)[::-1][:5]
    print('Cosine Similarity Metrics:')
    for i in cosine:
        f.write(label[i])
        f.write('\t')
        print(label[i], end='; ')
    f.write('\n')

    print('\n')

    dot = [np.dot(x,q) for x in data]
    metric = sorted(dot,reverse=True)[:5]
    f.write('Dot Product Metrics:')
    f.write(str(metric))
    f.write('\n')

    dot = np.argsort(dot)[::-1][:5]
    print('Dot Product Metrics:')
    for i in dot:
        f.write(label[i])
        f.write('\t')
        print(label[i], end='; ')


    f.close()
    print('\n' + '----'*30)

count = 0
for q in query:
    print("Thong tin truy van: " + anh[count])
    print(q)
    Metrics(q,data,count+1)
    count += 1
