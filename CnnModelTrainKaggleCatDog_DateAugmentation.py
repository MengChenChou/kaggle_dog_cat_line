# --coding:utf-8--
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

train_dir = 'C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/train'
test_dir = 'C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/test'
validation_dir = 'C:/Users/User/PycharmProjects/ai_deep/kaggle_cat_dog/kagglecatdog/validation'

model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                input_shape=(150,150,3),
                 activation='relu'
                ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#model.add(Dense(2,activation='softmax'))
# 顯示出模型摘要
model.summary()

#estop = EarlyStopping(monitor='val_loss', patience=3)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc'] )

# 資料增強增加學習樣本
train_datagen =  ImageDataGenerator(
  rescale=1./255, #指定將影象像素縮放到0~1之間
  rotation_range=45, # 角度值，0~180，影象旋轉
  width_shift_range=0.2, # 水平平移，相對總寬度的比例
  height_shift_range=0.2, # 垂直平移，相對總高度的比例
  shear_range=0.2, # 隨機傾斜角度
  zoom_range=0.2, # 隨機縮放範圍
  horizontal_flip=True,# 一半影象水平翻轉
  fill_mode = 'nearest' #產生新的影像若有出現空白處，「以最接近的像素」填補像素
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 訓練資料與測試資料  #分類超過兩類 使用categorical, 若分類只有兩類使用binary
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',)

# 使用批量生成器 模型模型
H = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size, #一共訓練100次
    epochs=30, #一共訓練回合
    validation_data=validation_generator,
    validation_steps=50)

model.save('model_CnnModelTrainKaggleCatDog_DateAugmentation.h5')

# 顯示acc學習結果
accuracy = H.history['acc']
val_accuracy = H.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
loss = H.history['loss']
val_loss = H.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()