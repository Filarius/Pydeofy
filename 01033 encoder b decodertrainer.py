import numpy as np

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Flatten,MaxPool2D,Concatenate,UpSampling2D,Dropout,LeakyReLU
from keras.layers import Subtract,Add
from keras.optimizers import Adam
import keras.optimizers as optims
from keras.metrics import Accuracy
from keras.activations import relu,sigmoid,tanh,softmax
import keras.activations as act
from keras.losses import mse,categorical_crossentropy
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
frame_cnt = 200#seconds*fps
bits_per_input = 8
samples_per_frame = w*h//64
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
        name = ('weights%04d.hdf5') % self.epoch
        self.model.save_weights(name)
        #name = ('weights%04d_1.hdf5') % self.epoch
        #self.model.save_weights(name)
    self.epoch += 1


def make_model():
    # encoder start
    enc_inps = [Input((2**bits_per_input,), name='enc_in{0}'.format(i)) for i in range(9 * 3)]
    enc_pipe = []
    enc_pipe.append(Concatenate(name='enc_cnc'))
    #enc_pipe.append(Flatten(name='enc_flat'))
    enc_pipe.append(Dense(64 * 2, activation=relu, name='enc_d1'))
    enc_pipe.append(Dense(8 * 8, activation=relu, name='enc_d3'))
    enc_pipe.append(Reshape((8, 8, 1), name='enc_r'))
    enc_lst = []
    for i in range(9*2):
        x = inp = [enc_inps[i], enc_inps[i+9]]
        for layer in enc_pipe:
            x = layer(x)
        enc_lst.append((inp, x))
    encoder = Model(inputs=enc_lst[0][0], outputs=enc_lst[0][1], name='encoder')
    enc_inputs = [enc_lst[x][0] for x in range(18)]
    enc_outputs = [enc_lst[x][1] for x in range(18)]
    # print(encoder.summary())
    # encoder end

    # decoder start
    decoder_input = x = Input((8, 8), name='dec_in')
    #x = Reshape((64,), name='dec_r')(x)
    x = Flatten(name='dec_f')(x)
    x = Dense(64 * 2, name='dec_d12', activation=relu)(x)
    x = Dense(64 * 1, name='dec_d2', activation=relu)(x)
    decoder_output = x = Dense(256, name='dec_d3', activation=softmax)(x)
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')

    autoencoder = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs), name='decoder')

    #print(decoder.summary())
    # decoder end
    '''
    # autoencoder start
    x = enc_inps
    y = enc_outputs
    y = noiser(y)
    y = decoder(y)
    autoencoder = Model(inputs=x, outputs=y, name='autoencoder')
    #print(autoencoder.summary())
    '''

    result = {'e': encoder,
              'd': decoder,
              #'n': noiser,
              'a': autoencoder
             }
    return result


models = make_model()
#models['d'].load_weights('decoder 20191029 v2.hdf5')
models['d'].trainable = False
optim = optims.Adam(learning_rate=0.0003)
models['a'].compile(optimizer=optim,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
models['d'].load_weights('decoder_weights0001.hdf5')
#models['e'].load_weights('encoder 20191029 v2.hdf5')
models['e'].load_weights('encoder weights0001.hdf5')



from ffmpegwrapper import ffmpeg

#path = 'ffmpeg -y -i testt.mp4 -f image2pipe -pix_fmt gray -c:v rawvideo -'
#path = 'ffmpeg -y -i test.mp4 3.mp4'.split(" ")

#ff = ffmpeg(path,use_stdout=True)
#ff = ffmpeg(path,print_to_console=True)
#ff.start()

'''
def enc_samples(cnt):
  global bits_per_input
  import random
  size = 2**bits_per_input
  mem = None
  x = np.zeros(shape=(cnt, size))
  max = 0
  for i in range(1,cnt):
          k = random.randint(0,size-1)
          if max < k:
              max = k
          x[i,k] = 1.0
  return x

def frame_samples_gen(cnt):
    ENC_CNT = 1024
    #encs = enc_samples(ENC_CNT)
    import pickle
    with open('random.dat',mode='rb') as f:
        encs = pickle.load(f)
    ind = 1
    dx = 1
    frame_now = []
    frame_before = [encs[0] for _ in range(samples_per_frame)]
    for k in range(cnt):
        for i in range(samples_per_frame):
            frame_now.append(encs[ind])
            ind += dx
            if ind == ENC_CNT:
                ind = ENC_CNT - 2
                dx = -1
            else:
                if ind == 0:
                    ind = 1
                    dx = 1
        yield [frame_now,frame_before]
        frame_before = frame_now
        frame_now = []
'''
def enc_samples(cnt):
  global bits_per_input
  import random
  size = 2**bits_per_input
  mem = None
  x = np.zeros(shape=(cnt, size))
  for i in range(0,cnt):
          k = random.randint(0,size-1)
          x[i,k] = 1.0
  return x

def frame_samples_gen(cnt):
    ENC_CNT = 1024
    #encs = enc_samples(ENC_CNT)
    import pickle
    with open('random.dat',mode='rb') as f:
        encs = pickle.load(f)
    ind = 1
    dx = 1
    frame_now = np.empty((samples_per_frame,2**bits_per_input),dtype=np.float)
    frame_before = np.zeros((samples_per_frame, 2**bits_per_input), dtype=np.float)
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
    import pickle
    f = open('random.db','wb')
    np.random.seed(2)
    samples_front = np.random.randint(2,size= (samples_per_frame,bits_per_input))
    samples_back = np.zeros(shape=(samples_per_frame,bits_per_input), dtype=np.int32)
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



samples = np.zeros((samples_per_frame,8,8),dtype=np.float)



cnt = 0
for x in frame_samples_gen(frame_cnt):
    if cnt == 0:
        cnt += 1
        continue

    #predict_test = models['d'].predict(x)
    #predict_test = np.clip(predict_test, a_min=0.0, a_max=1.0)
#    frame = ff.readout(h*w)
#    frame = frame.reshape((h, w))
    #print(frame.shape)
    models['a'].fit(x,
                    x[0],
                    batch_size=1024,
                    epochs=1,
                    callbacks=[WeightsSaver(1)])
    models['e'].save_weights('encoder weights0001.hdf5')
    #frame_to_samples(frame,samples)























