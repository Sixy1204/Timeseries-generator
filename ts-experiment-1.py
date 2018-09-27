# -*- coding: utf-8 -*-
# Inspecting the data of the Jena weather dataset
import os
data_dir = 'F:\Python\senior-year-research'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

#Parsing the data
#原资料---->2D
import numpy as np
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# Plotting the temperature timeseries
 
from matplotlib import pyplot as plt
temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
#Plotting the first 10 days of the temperature timeseries
plt.plot(range(1440), temp[:1440])

# Normalizing the data标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

#Generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:  #if shuffle==TRUE 随机取一点向前看历史数据，再预测向后delay=144那一点
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
            #在min_index和max_index之间随机选batch_size=128个点（index）赋值给rows
        
        else: #if shuffle==FALSE 按时间窗口移动
            if i + batch_size >= max_index: #防错机制
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            #从i1=min_index+lookback开始按时间顺序向后移动(1~128个点)
            #每一点向前看历史数据，再预测向后delay=144那一点
            i += len(rows) #更新i
        
        #2D---->3D
        samples = np.zeros((len(rows), lookback // step, data.shape[-1])) #(128,240,14)
        #samples(128笔抽样资料, 240个历史时间点, 14个特征) 注解(3D)
        
        targets = np.zeros((len(rows),))
        #每笔抽样资料对应的预测目标y，总共128个y 注解(1D)
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step) #回顾一次历史资料按时间顺序的index(240个)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
    
#Preparing the training, validation, and test generators
lookback = 1440 #回溯10天前的资料
step = 6 #每一小时取样一次
delay = 144 #预测一天后的资料
batch_size = 128
train_gen = generator(float_data,lookback=lookback,delay=delay,min_index=0,
                      max_index=3000,shuffle=True,step=step,batch_size=batch_size)
val_gen = generator(float_data,lookback=lookback,delay=delay,
                    min_index=3001,max_index=5000,step=step,batch_size=batch_size)
test_gen = generator(float_data,lookback=lookback,delay=delay,
                     min_index=5001,max_index=None,step=step,batch_size=batch_size)
val_steps = (5000 - 3001 - lookback)
test_steps = (len(float_data) - 5001 - lookback)
#a=next(train_gen)
#b=next(val_gen)
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1] #近似认为时间点N(顺序移动的最后一个点)和N+1(预测)的温度一样
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    
evaluate_naive_method()

#Converting the MAE back to a Celsius error
celsius_mae = 0.29 * std[1]

#Training and evaluating a densely connected model
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

