import numpy as np

from keras.models import Model
from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Reshape,BatchNormalization,Flatten,MaxPool2D,Concatenate,UpSampling2D,Dropout,LeakyReLU
from keras.layers import Subtract,Add,GaussianNoise,Lambda
from keras.optimizers import Adam
import keras.optimizers as optims
from keras.metrics import Accuracy,binary_accuracy
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
bits_per_input = 16
ones_count = 2

from binary_encoder import BinaryCoder
coder = BinaryCoder(2**bits_per_input)
width = coder.get_width()

#from sklearn.preprocessing.label import LabelBinarizer
#lcoder = LabelBinarizer(sparse_output=True)
#width = 2**bits_per_input

#lcoder.fit(range(width))
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


def Sampler(x):
    mean = K.mean(x)
    std = K.abs(mean-0.5)*1
    #x = GaussianNoise(std)(x)
    x = x
    #x = K.clip(x,0.0,1.0)
    #x = K.clip(x,0,1)
    return Lambda(lambda x:K.clip(x,0,1))(x)

def make_model():
    # encoder start
    enc_inps = [Input((width,), name='enc_in{0}'.format(i)) for i in range(9 * 3)]
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
        x = Sampler(x)
        enc_lst.append((inp, x))
    encoder = Model(inputs=enc_lst[0][0], outputs=enc_lst[0][1], name='encoder')
    enc_inputs = [enc_lst[x][0] for x in range(18)]
    enc_outputs = [enc_lst[x][1] for x in range(18)]

    # print(encoder.summary())
    # encoder end

    # noiser start
    noiser_inputs = [Input((8, 8), name='noi_in{0}'.format(i)) for i in range(9 * 2)]

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
    #print(noiser.summary())
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

    # autoencoder start
    x = enc_inps
    y = enc_outputs
    y = noiser(y)
    y = Sampler(y)
    y = decoder(y)
    autoencoder = Model(inputs=x, outputs=y, name='autoencoder')
    #print(autoencoder.summary())

    result = {'e': encoder,
              'd': decoder,
              'n': noiser,
              'a': autoencoder}
    return result



models = make_model()
# print(model.summary())

def bit_chk(x,y):
   # print(x.shape)
    return K.abs(K.mean((x-y),-1)) + 2 * K.std((x-y),-1)

assert bits_per_input is not None
models = make_model()
optim = optims.Adam(learning_rate=0.001)
models['n'].trainable = False
#models['d'].trainable = False
models['a'].compile(optimizer=optim,
              loss=binary_crossentropy,#bit_mean_abs,#'mae',
              #metrics=[bit_max,bit_mean,bit_std,bit_chk])
              metrics=[binary_accuracy])
models['a'].load_weights('a.hdf5')
models['n'].load_weights('n.hdf5')
models['e'].save_weights('e.hdf5')
models['d'].save_weights('d.hdf5')

#for layer in models['a'].layers:
#  print(layer.name,' ',str(layer.trainable))
from scipy.sparse import csr_matrix
def enc_dec_train_get(cnt):
  global bits_per_input
  import random
  size = width

  #x = np.zeros(shape=(cnt, 9*3, size))

  x = np.random.randint(2**bits_per_input,size=(cnt+3)*9,dtype=coder.get_dtype())
  #x = coder.encode(x)
  x = coder.encode(x)
  '''
  for i in range(cnt):
      if cnt <= 2:
          for j in range(9*3):
              k = random.randint(0,size-1)
              x[i,j,k] = 1.0
      else:
          for j in range(9):
              k = random.randint(0,size-1)
              x[i,j,k] = 1.0
          x[i, 9:18, :] = x[i - 1, 0:9, :]
          x[i, 18:27, :] = x[i - 2, 0:9, :]
  '''
  #xx = [x[3:,i,:] for i in range(9*3)]
  #y = x[3:,4,:]
  xx = [x[i:-27+i:9] for i in range(9*3)]
  y = xx[4]
  return (xx,y)

x,y = enc_dec_train_get(1024*8*2)
print("t1")
#xv,yv = enc_dec_train_get(1024)
print("t2")
batch = 512
train_size = 1028*16

graph =  True
graph =  not True
if graph:
    import matplotlib.pyplot as plt
    k=-1
    for i in range(8000):
        for j in range(2**bits_per_input):
            k += 1

            xq = enc_dec_train_get(1000)
            xq = [xq[0][4],xq[0][4+9]]
            if k%(8*8) ==0:
                if k > 0:
                    plt.show()
                    #exit(0)
                fig, axes = plt.subplots(8,8)
            yq = models['e'].predict(xq)[0,:,:,0]
            print(np.min(yq),np.max(yq),np.mean(yq))
            axes[(k%(8*8))//8,k%8].imshow(yq,cmap='gray')
            axes[(k % (8 * 8)) // 8, k % 8].axis('off')
            #fig.set_figwidth(2)
            #fig.set_figheight(2)




if not graph:
 while True:
  x, y = enc_dec_train_get(train_size)
  models['a'].fit(x,y,batch_size=batch,epochs=1,
              callbacks=[WeightsSaver(1)]#,tb]
              #,validation_data=((xv,yv))
              ,verbose=True)
