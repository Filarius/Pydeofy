import numpy as np

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Flatten,MaxPool2D,Concatenate,UpSampling2D,Dropout,LeakyReLU
from keras.layers import Subtract,Add
from keras.optimizers import Adam
import keras.optimizers as optims
from keras.metrics import Accuracy
from keras.activations import relu,sigmoid,tanh
import keras.activations as act
from keras.losses import mse
import keras.losses
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback,TensorBoard
import keras.backend as K



w = 1920
h = 1080
h_t = 864 # 1080*0.8
fps = 60
seconds = 60
batch = 1

def pix_max(x,y):
    return K.max(K.abs(x-y)*255,-1)
def pix_mean(x,y):
    return K.mean(K.abs(x-y)*255,-1)
def pix_std(x,y):
    return K.std(K.abs(x-y)*255,-1)
def bit_max(x,y):
    return K.max(K.abs(x-y)*2,-1)
def bit_mean(x,y):
    return K.mean(K.abs(x-y)*2,-1)
def bit_std(x,y):
    return K.std(K.abs(x-y)>0.5,-1)

class WeightsSaver(Callback):
  def __init__(self, N):
    self.N = N
    self.epoch = 0

  def on_epoch_end(self, epoch, logs={}):
    if self.epoch % self.N == 0:
        self.model
        print('saving ',epoch)
        name = ('e_weights%04d.hdf5') % self.epoch
        self.model.save_weights(name)
        #name = ('weights%04d_1.hdf5') % self.epoch
        #self.model.save_weights(name)
    self.epoch += 1


import cv2


def train_gen_colab_2(batch, skip, pair_num):
    goods = []
    goods_b = []
    bads = []

    capgood = cv2.VideoCapture('bad{0}.mp4'.format(pair_num))
    capbad = cv2.VideoCapture('bad{0}.mp4'.format(pair_num+1))

    i = -1
    print(capgood.isOpened())
    print(capbad.isOpened())
    frame_gb = None
    frame_bb = None
    while (capgood.isOpened() and capbad.isOpened()):
        retg, frame_g = capgood.read()
        retb, frame_b = capbad.read()
        if not (retg and retb):
            break

        i += 1
        print("frame #", i)
        frame_g = frame_g[:, :, 0]
        frame_b = frame_b[:, :, 0]

        if i < 1:
            frame_gb = frame_g
            continue
        if i < skip:
            continue

        # frame_g = frame_g.reshape(h, w)
        # frame_b = frame_b.reshape(h, w)
        samples_idx = 0
        for x in range(8, w - 16, 8):
            # print('x = ',x)
            for y in range(8, h - 16, 8):

                good = frame_g[y - 8:y + 16, x - 8:x + 16].astype(np.float) / 255

                good_b = frame_gb[y - 8:y + 16, x - 8:x + 16].astype(np.float) / 255
                g = []
                gb = []
                for q in range(3):
                    for z in range(3):
                        g.append(good[8 * q:8 + 8 * q, 8 * z:8 + 8 * z])
                        gb.append(good_b[8 * q:8 + 8 * q, 8 * z:8 + 8 * z])
                goods.append(g + gb)
                # goods_b.append(good_b)

                bad = frame_b[y:y + 8, x:x + 8]
                bad = bad.astype(np.float)[:, :] / 255
                bads.append(bad)
                if len(bads) == batch:
                    # yield ([np.array(goods),np.array(goods_b)], np.array(bads))
                    # goods_transpose = list(map(np.array,zip(*goods)))
                    # print(goods_traspose[0])
                    goods = list(map(list, zip(*goods)))  # list transpose
                    while True:
                        yield (goods, np.array(bads))

                    bads = []
                    goods = []
                    goods_b = []
        frame_gb = frame_g

    capgood.release()
    capbad.release()


bits_per_input = 8

def make_model():
    # encoder start
    enc_inps = [Input((bits_per_input,), name='enc_in{0}'.format(i)) for i in range(9 * 2)]
    enc_pipe = []
    enc_pipe.append(Dense(64 * 8, activation=relu, name='enc_d1'))

    enc_pipe.append(Dense(64 * 4, activation=relu, name='enc_d22'))
    # enc_pipe.append(Dropout(rate=0.2))
    enc_pipe.append(Dense(64 * 2, activation=relu, name='enc_d2'))
    # enc_pipe.append(Dropout(rate=0.2))
    enc_pipe.append(Dense(8 * 8, activation=relu, name='enc_d3'))
    enc_pipe.append(Reshape((8, 8, 1), name='enc_r'))


    enc_lst = []
    for inp in enc_inps:
        x = inp
        for layer in enc_pipe:
            x = layer(x)
        enc_lst.append((inp, x))
    encoder = Model(inputs=enc_lst[0][0], outputs=enc_lst[0][1], name='encoder')
    enc_inputs = [enc_lst[x][0] for x in range(18)]
    enc_outputs = [enc_lst[x][1] for x in range(18)]

    # print(encoder.summary())
    # encoder end

    # noiser start
    noiser_inputs = [Input((8, 8), name='noi_in{0}'.format(i)) for i in range(9 * 2)]
    '''
    pipe = []
    pipe.append(Reshape((8, 8, 1), name='noi_r'))
    pipe.append(Conv2D(8, 3, strides=1, padding='same', activation=relu, name='noi_c01'))
    pipe.append(BatchNormalization())
    pipe.append(MaxPool2D((2, 2), strides=2, padding='same'))
    pipe.append(Conv2D(16, 3, strides=1, padding='same', activation=relu, name='noi_c02'))
    pipe.append(BatchNormalization())
    pipe.append(MaxPool2D((2, 2), strides=2, padding='same'))
    pipe.append(Conv2D(32, 3, strides=1, padding='same', activation=relu, name='noi_c03'))
    pipe.append(BatchNormalization())
    pipe.append(MaxPool2D((2, 2), strides=2, padding='same'))
    pipe_outputs = []
    for input in noiser_inputs:
        x = input
        for layer in pipe:
            x = layer(x)
        pipe_outputs.append(x)
    '''
    # x = [Concatenate(axis=1, name='enc_c1_{y}')(noiser_inps[y:y+3]) for y in range(0,18,3))
    x = Concatenate(axis=1, name='noi_c')(noiser_inputs)
    x = Flatten(name='noi_f')(x)
    x = Dense(64 * 16, name='noi_d1', activation=relu)(x)
    x = Dense(64 * 8, name='noi_d2', activation=relu)(x)
    #x = Dense(64 * 2,name='noi_d3',activation=relu)(x)
    #x = Dense(64 * 2, name='noi_d31', activation=relu)(x)
    x = Dense(8 * 8, name='noi_d4', activation=sigmoid)(x)
    noiser_output = x = Reshape((8, 8), name='noi_r')(x)
    noiser = Model(inputs=noiser_inputs, outputs=noiser_output, name='noiser')
    print(noiser.summary())
    # noiser end

    # decoder start
    decoder_input = x = Input((8, 8), name='dec_in')
    x = Reshape((8, 8, 1), name='dec_r')(x)

    x = Flatten(name='dec_f')(x)
    x = Dense(8 * 8 * 8, name='dec_d1', activation=relu)(x)
    # x = Dropout(rate=0.2)(x)
    x = Dense(64 * 4, name='dec_d11', activation=relu)(x)
    # x = Dropout(rate=0.2)(x)
    x = Dense(64 * 2, name='dec_d12', activation=relu)(x)
    x = Dense(64 * 1, name='dec_d2', activation=relu)(x)
    decoder_output = x = Dense(bits_per_input, name='dec_d3', activation=relu)(x)
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
    # print(decoder.summary())
    # decoder end

    # autoencoder start
    x = enc_inputs
    y = enc_outputs
    y = noiser(y)
    y = decoder(y)
    autoencoder = Model(inputs=x, outputs=y, name='autoencoder')
    # print(autoencoder.summary())

    result = {'e': encoder,
              'd': decoder,
              'n': noiser,
              'a': autoencoder}
    return result

models = make_model()

optim = optims.Adam(learning_rate=0.00005)
models['n'].compile(optimizer=optim,
              loss='mae',
              metrics=[pix_max,pix_mean,pix_std])

models['n'].load_weights('e_weights0000.hdf5')
from random import randint
max_pair = 17
max_frame = 50
while True:
    pair_train = randint(0,max_pair)
    pair_validate = randint(1,max_pair)
    start_frame_train = randint(1,max_frame)
    start_frame_validate = 0*randint(1,max_frame)
    print('start new training')
    print(pair_train,pair_validate,start_frame_train,start_frame_validate)
    for x in train_gen_colab_2(1024 * 32*1,
                                      skip=start_frame_train,
                                      pair_num=pair_train):
        batch = 1024
        models['n'].fit(x[0], x[1], batch_size=batch, epochs=4,
                        callbacks=[WeightsSaver(1 )]  # ,tb]
                       # , validation_data=((y[0], y[1]))
                        , verbose=True)
        break
