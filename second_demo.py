import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation,Dense,Dropout
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
from  keras.utils import np_utils
from keras import initializers
from keras.utils.vis_utils import plot_model
# def init_weights(shape,name=None):#是否要做
#     return initializers.normal(shape,scale=0.01,name=name)
#Using TensorFlow backend.
def load_data():
    #载入数据
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    print('X_train original shape:', x_train.shape)
    number=10000
    #数据处理
    x_train=x_train[0:number]
    y_train=y_train[0:number]
    x_train=x_train.reshape(number,28*28)
    x_test=x_test.reshape(x_test.shape[0],28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)
    x_train=x_train
    x_test=x_test
    #normalization非常关键
    x_train=x_train/255
    x_test=x_test/255
    # x_test=np.random.normal(x_test)#加了噪声
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test)=load_data()
#建立模型
model=Sequential()

model.add(Dense(input_dim=28*28,units=500,activation='relu'))
# model.add(Dense(input_dim=28*28,output_dim=500)) #输入层28*28，也就是图片，第一个输出层500个神经元
# model.add(Activation('sigmoid'))#激活函数
# model.add(Dropout(0.7))#进行Drooupt
model.add(Dense(units=500,activation='relu'))
# model.add(Dense(output_dim=500))#第二个输出层，500个神经元
# model.add(Activation('sigmoid'))
# model.add(Dropout(0.7))#进行Drooupt
model.add(Dense(units=10,activation='softmax'))
# model.add(Dense(output_dim=10))#最后输出层，10维
# model.add(Activation('softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',#损失函数进行评估
              optimizer='adam',metrics=['accuracy'])#优化函数

model.fit(x_train,y_train,batch_size=100,nb_epoch=20)#Image,label,100个eample放在batch，每个batch重复20次

#在trainning data 进行验证
result=model.evaluate(x_train,y_train,batch_size=10000)
print('\nTrain acc',result[1])
#testing data
score=model.evaluate(x_test,y_test,batch_size=10000)
print('Total loss on Testing Set:',score[0])
print('\nAccuracy of  Testing Set: ',score[1])

# result=model.predict(x_test)



