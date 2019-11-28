import numpy as np

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Flatten,MaxPool2D,Concatenate,UpSampling2D,Dropout,LeakyReLU
from keras.layers import Subtract,Add,Lambda
from keras.optimizers import Adam
import keras.optimizers as optims
from keras.metrics import Accuracy
from keras.activations import relu,sigmoid,tanh,softmax
import keras.activations as act
from keras.losses import mse,categorical_crossentropy,binary_crossentropy
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
bits_per_input = 8
ones_count = 1
samples_per_frame = w*h//64
frame_cnt = 40

import multihotencoding
coder = multihotencoding.MultiHotEncoding(2**bits_per_input,ones_count=ones_count)
width = coder.get_width()
#функции точности, показывают отклонение от идеала - максимальное, среднее, и разброс
def pix_max(x,y):
    return K.max(K.abs(x-y)*255,-1)
def pix_mean(x,y):
    return K.mean(K.abs(x-y)*255,-1)
def pix_std(x,y):
    return K.std(K.abs(x-y)*255,-1)
def bit_max(x,y):
    return K.max(K.abs(x-y),-1)
def bit_mean(x,y):
    return K.mean((x-y),-1)
def bit_mean_abs(x,y):
    return K.mean(K.abs((x-y)),-1)
def bit_std(x,y):
    return K.std((x-y),-1)

class WeightsSaver(Callback):
  def __init__(self, N):
    self.N = N
    self.epoch = 0

  def on_epoch_end(self, epoch, logs={}):
    if self.epoch % self.N == 0:
        self.model
        print('saving ',epoch)
        #name = ('a%04d.hdf5') % self.epoch
        name = ('a.hdf5')
        self.model.save_weights(name)
        #name = ('weights%04d_1.hdf5') % self.epoch
        #self.model.save_weights(name)
    self.epoch += 1




def make_model():
    # encoder start
    enc_inps = [Input((width,), name='enc_in{0}'.format(i)) for i in range(9 * 3)]
    enc_pipe = []
    enc_pipe.append(Concatenate(name='enc_cnc'))
    #enc_pipe.append(Flatten(name='enc_flat'))
    enc_pipe.append(Dense(64 * 2, activation=relu, name='enc_d1'))
    #enc_pipe.append(Dense(64 * 2, activation=relu, name='enc_d2'))
    enc_pipe.append(Dense(8 * 8, activation=relu, name='enc_d3'))
    enc_pipe.append(Reshape((8, 8, 1), name='enc_r'))
    enc_lst = []
    for i in range(9*2):
        x = inp = [enc_inps[i], enc_inps[i+9]]
        for layer in enc_pipe:
            x = layer(x)
        x = Lambda(lambda x:K.clip(x,0,1))(x)
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
    n_output = Dense(8 * 8, name='noi_d4', activation=sigmoid)
    x = n_output(x)
    noiser_output = x = Reshape((8, 8), name='noi_r')(x)
    noiser = Model(inputs=noiser_inputs, outputs=noiser_output, name='noiser')
    print(noiser.summary())
    # noiser end

    # decoder start
    decoder_input = x = Input((8, 8), name='dec_in')
    #x = Reshape((64,), name='dec_r')(x)
    x = Flatten(name='dec_f')(x)
    x = Dense(64 * 2, name='dec_d12', activation=relu)(x)
    x = Dense(64 * 1, name='dec_d2', activation=relu)(x)
    decoder_output = x = Dense(width, name='dec_d3', activation=sigmoid)(x)
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
    #print(decoder.summary())
    # decoder end

    def sampler(x):
        #mean = K.mean(x)
        #std = K.abs(mean - 0.5)*300
        #x = GaussianNoise(std)(x)
        x = K.clip(x,0,1)
        return x



    # autoencoder start
    x = enc_inps
    y = enc_outputs
    y = noiser(y)
    #y = Lambda(sampler)(y)
    y = decoder(y)
    autoencoder = Model(inputs=x, outputs=y, name='autoencoder')
    #print(autoencoder.summary())


    result = {'e': encoder,
              'd': decoder,
              'n': noiser,
              'a': autoencoder}

    #K.abs(mean-0.5) + K.abs(std-0.2)

    return result

models = make_model()
#models['d'].load_weights('decoder 20191029 v2.hdf5')
models['d'].load_weights('d.hdf5')
from ffmpegwrapper import ffmpeg

path = 'ffmpeg -y -i testt.mp4 -f image2pipe -pix_fmt gray -c:v rawvideo -'
#path = 'ffmpeg -y -i test.mp4 3.mp4'.split(" ")

ff = ffmpeg(path,use_stdout=True)
#ff = ffmpeg(path,print_to_console=True)
ff.start()

def enc_samples(cnt):
  global bits_per_input
  import random
  x = np.random.randint(256,size=cnt)
  ret = coder.encode(x)
  #x = np.zeros(shape=(cnt, size))
  #for i in range(0,cnt):
  #        k = random.randint(0,size-1)
  #        x[i,k] = 1.0
  return ret

def frame_samples_gen(cnt):
    ENC_CNT = 1024*(32+8)
    #encs = enc_samples(ENC_CNT)
    import pickle
    with open('random.dat',mode='rb') as f:
        encs = pickle.load(f)
    ind = 1
    dx = 1
    frame_now = np.empty((samples_per_frame,width),dtype=np.float)
    frame_before = np.zeros((samples_per_frame, width), dtype=np.float)

    for k in range(cnt):
        for i in range(samples_per_frame):
            frame_now[i,:] = encs[ind]
            ind += dx
            if ind == ENC_CNT:
                ind = ENC_CNT - 2
                dx = -1
            else:
                if ind == 0:
                    dx = 1
        yield [frame_now,frame_before]
        temp = frame_before
        frame_before = frame_now
        frame_now = temp # UNSAFE ! BEWARE


def samples_generator(cnt):
    raise Exception()
    import pickle
    f = open('random.db','wb')
    np.random.seed(2)
    samples_front = np.random.randint(2,size= (samples_per_frame,width))
    samples_back = np.zeros(shape=(samples_per_frame,width), dtype=np.int32)
    for i in range(cnt):
        x = [samples_front,samples_back]
        pickle.dump(x,f)
        yield x
        samples_back = samples_front
        samples_front = np.random.randint(2,size= (samples_per_frame,bits_per_input))
    f.close()



#frame = np.zeros((h, w), dtype=np.uint8)



from numba import jit
@jit(nopython=True)
def frame_to_samples(frame,samples):
    #predict = predict*255
    #predict = predict.astype(np.uint8)
    for i in range((h // 8)):
        for j in range((w // 8)):
            samples[i * (w // 8) + j, :, :] = frame[i * 8:i * 8 + 8, j * 8:j * 8 + 8]/255


cnt = 0
samples = np.zeros((samples_per_frame,8,8),dtype=np.float)



bad_luck_row = 0
last_luck_row = 0
for x in frame_samples_gen(frame_cnt):
    #predict_test = models['d'].predict(x)
    #predict_test = np.clip(predict_test, a_min=0.0, a_max=1.0)
    frame = ff.readout(h*w)
    frame = frame.reshape((h, w))
    #print(frame.shape)

    frame_to_samples(frame,samples)
    y = models['d'].predict(samples)
    r0 = coder.decode_legacy(x[0])
    r1 = coder.decode_legacy(y)

    cnt = 0
    bads = 0
    rets = 0
    last_luck_row = 0
    for i in range(samples_per_frame):
        x0 = r0[i]
        x1 = r1[i]
        if x0==x1:
            cnt += 1
            last_luck_row = 0
        else:
            last_luck_row += 1
            if bad_luck_row < last_luck_row:
                bad_luck_row =last_luck_row
            bads += 1
            arg = np.argmax(y[i])
            loc_ret = 0


    print(cnt / samples_per_frame, " unlucky ",last_luck_row, bad_luck_row)




















