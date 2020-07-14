# --coding:utf-8--

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

train_dir = 'C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/train'
test_dir = 'C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/test'
validation_dir = 'C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/validation'

train_datagen =  ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir )

print('='*30)
print('訓練的分類：',train_generator.class_indices)
print('='*30)

labels = train_generator.class_indices

#將分類做成字典方便查詢
labels = dict((v,k) for k,v in labels.items())
print(labels)

# 載入模型
model = load_model('model_CnnModelTrainKaggleCatDog_DateAugmentation.h5')

# 將圖片轉為待測數據
def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
    except Exception as e:
        print(img_path,e)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# 隨機輸入一個待測圖片
filename = "C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/test/cat/cat.1684.jpg"

plt.figure()
im = Image.open(filename)
im_list = np.asarray(im)
plt.title("predict")
plt.axis("off")
plt.imshow(im_list)
plt.show()

img = read_image(filename)
pred = model.predict(img)[0]
print('辨識結果:',labels[pred[0]])