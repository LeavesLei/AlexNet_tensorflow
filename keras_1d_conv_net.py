import keras
import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np

def convert_txt_to_np_with_label(file, label):
    """
    convert .txt file to numpy matrix and add label
    return x: numpy matrix with 3 dimension (-1, 2, 16)
    return y: label with 1 dimension
    """
    a=[]
    with open(file, 'r') as f:
        data = f.readlines()  
        for line in data:
            if len(line)>11:
                b = line[:-1].split(' ')
                c = []
                for i in b:
                    if i != '':
                        c.append(i)
                c = map(float, c)
                a.append(list(c))
    x = np.array(a).reshape((-1, 2, 16))
    y = np.array([label] * x.shape[0])
    return x, y

# read dataset
x0, y0 = convert_txt_to_np_with_label('IQdanyin.txt', 0)
x1, y1 = convert_txt_to_np_with_label('IQduoyin.txt', 1)
x2, y2 = convert_txt_to_np_with_label('IQbufen.txt', 2)
x3, y3 = convert_txt_to_np_with_label('IQxianxing.txt', 3)

# gather dataset
num_classes = 4
x = np.vstack((x0, x1, x2, x3))
y = np.hstack((y0, y1, y2, y3))
y = np_utils.to_categorical(y, num_classes)

# shuffle
index=np.arange(x.shape[0])
np.random.shuffle(index)

x = x[index, :, :]
y = y[index, :]

# model parameter setting
epoch = 20
kernel_size = 5
batch_size = 8
input_shape = (2, 16)

# build model
model = Sequential()
model.add(Conv1D(32, kernel_size=kernel_size, padding='same', activation='relu', input_shape=input_shape, data_format="channels_first"))
model.add(MaxPooling1D(pool_size=2, padding='same', strides=1, data_format='channels_first'))
model.add(Conv1D(64, kernel_size, padding='same', activation='relu', data_format="channels_first"))
model.add(MaxPooling1D(pool_size=2, padding='same', strides=1, data_format='channels_first'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# model setting
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# training model
history = model.fit(x, y, validation_split=0.3, epochs=epoch, batch_size=batch_size)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# accuracy plot 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model accuracy" )
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.close()
# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('model_loss.png')
plt.close()
