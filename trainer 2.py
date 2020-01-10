from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.adamw import AdamW
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from subprocess import PIPE, Popen
from threading  import Thread
from queue import Queue
from time import sleep
import numba
import zfec

fec_enc = zfec.easyfec.Encoder(98,100)
fec_dec = zfec.easyfec.Decoder(98,100)

from ffmpegwrapper import ffmpeg
from binary_encoder import BinaryCoder



m = 7
w = 16*5*m
h = 16*5*m

#w = 1920
#h = 1080

h_t = 864 # 1080*0.8
fps = 60
seconds = 60
batch = 1
bits_per_input = 8*4
bytes_per_input = bits_per_input // 8
samples_per_frame = w*h//64
bytes_per_frame = samples_per_frame*bytes_per_input

frame_cnt  = 100
batch_size = int(samples_per_frame*1)
batch_cnt = samples_per_frame//batch_size
epochs = 1
lr = 0.005
sleep_time = 0.4
noise_std = 100/255
features_mult = 20
learn = '0000'=='0000'
youtube_testing = '0000'=='000'
youtube_tuning = '0000'=='000'

coder = BinaryCoder(bits_per_input)
width = coder.get_width()
f1 = width*2
f2 = width//2
f3 = width//4

def enc_samples(cnt):
  global bits_per_input
  import random
  size = 2**bits_per_input
  mem = None
  x = np.random.randint(0,256,size=cnt*(bits_per_input//8),dtype=np.uint8)
  y = coder.encode(x)
  return y

def frame_samples_gen(cnt,renew=False):
    ENC_CNT = 1024*(100+16)
    import pickle
    if not renew:
        try:
            with open('random.dat',mode='rb') as f:
                encs = pickle.load(f)
            if encs.shape[1] != width:
                raise Exception()
        except:
            encs = enc_samples(ENC_CNT)
            with open('random.dat',mode='wb') as f:
                pickle.dump(encs,f)
    else:
        encs = enc_samples(ENC_CNT)
        with open('random.dat', mode='wb') as f:
            pickle.dump(encs, f)

    ind = 1
    dx = 1
    frame_now = np.empty((width, w//8, h//8), dtype=np.uint8)
    for k in range(cnt):
        for i in range((w//8)):
            for j in range((h // 8)):
                frame_now[:,i,j] = encs[ind,:]
                ind += dx
                if ind == ENC_CNT:
                    ind = ENC_CNT - 2
                    dx = -1
                else:
                    if ind == 0:
                        dx = 1
        yield frame_now

from numba import jit
@jit(nopython=True)
def frame_to_samples(frame,samples):
    for i in range((h // 8)):
        for j in range((w // 8)):
            samples[i * (w // 8) + j, :] = (frame[i * 8:i * 8 + 8, j * 8:j * 8 + 8] / 255).reshape(64)

@jit(nopython=True)
def frame_from_samples(frame,samples):
    samples = samples*255
    samples = samples.astype(np.uint8)
    for i in range((h // 8)):
        for j in range((w // 8)):
            frame[i * 8: i * 8 + 8, j * 8: j * 8 + 8] = samples[i * (w // 8) + j, :].reshape(8,8)

def pipe_to_queue(ff:ffmpeg, queue:Queue,cnt=10240):
    while True:
        #print('read!')
        a = ff.readout(cnt)
        if len(a)==0:
            sleep(0.1)
            continue
        #print('read! ', str(a.shape))
        queue.put(a)


class Encoder(nn.Module):
    def __init__(self, width):
        super(Encoder,self).__init__()
        self._width = width
        #self.ct1 = nn.ConvTranspose2d(in_channels=width, out_channels=features_mult, kernel_size=8, stride=8)

        self.ct1 = nn.ConvTranspose2d(in_channels=width, out_channels=f1, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=f1)
        self.ct2 = nn.ConvTranspose2d(in_channels=f1, out_channels=f2, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=f2)
        self.ct3 = nn.ConvTranspose2d(in_channels=f2, out_channels=f3, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=f3)
        self.ct4 = nn.ConvTranspose2d(in_channels=f3, out_channels=1, kernel_size=1,stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=1)
        #self.fc1 = nn.Linear(features_mult * 9, 64)
        #self.bn1 = nn.BatchNorm2d(num_features=f1)

    def forward(self,inp):
        x = inp[0].view(1, width, w//8, h//8)
        x = self.ct1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.ct2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.ct3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.ct4(x)
        x = self.bn4(x)
        x = torch.sigmoid(x)
        x= x.clamp(min=0.0,max=1.0)
        return x



class Decoder(nn.Module):
    def __init__(self, width):
        super(Decoder, self).__init__()

        self.ct1 = nn.Conv2d(in_channels=1, out_channels=f3, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=f3)
        self.ct2 = nn.Conv2d(in_channels=f3, out_channels=f2, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=f2)
        self.ct3 = nn.Conv2d(in_channels=f2, out_channels=f1, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=f1)
        self.ct4 = nn.Conv2d(in_channels=f1, out_channels=width, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(num_features=width)

    def forward(self, inp):
        x = inp[0].view(1, 1, w, h)
        x = self.ct1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.ct2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.ct3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.ct4(x)
        x = self.bn4(x)
        x = torch.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, width, noise_std=None):
        super(Autoencoder,self).__init__()
        self._noise_std = noise_std
        self._width = width
        self.decoder = Decoder(width)
        self.encoder = Encoder(width)

    def forward(self,inp):
        # samples, back_samples, back_frame_pixels_reencoded
        x = self.encoder([inp[0]])
        #x = x+torch.Tensor(inp[1])
        if self._noise_std is not None:
            x = x + self._noise_std*torch.randn_like(x)
        x = x.clamp(0.0,1.0)

        enc = x
        #x = x.view(w, h)
        #y = x - inp[2]
        x = self.decoder([x])
        return x,enc # predicted samples, reencoded front frame


def binary_accuracy(check, target):
    '''Binary accuracy metric'''
    check = check >= 0.5
    target = target >= 0.5
    ret = check.eq(target)
    ret = ret.sum().type(torch.float32)
    ret = ret / target.numel()
    return ret


if __name__=='__main__':

    ff_write = 'ffmpeg -loglevel quiet -f rawvideo -pix_fmt gray -s:v ' + str(w) + 'x' + str(h
             ) + ' -i - -pix_fmt yuv420p -c:v libx264 -preset ultrafast -crf 28 -g 50 -tune zerolatency -f h264 -'
    ff_write = ff_write.split(' ')
    ff_read = 'ffmpeg -loglevel quiet -probesize 32 -f h264 -i - -c:v rawvideo -pix_fmt gray -f image2pipe -'.split(' ')

    ff_write_file = 'ffmpeg -y -loglevel panic -f rawvideo -pix_fmt gray -s:v {0}x{1} -r 60 -i - -c:v libx264 -preset fast -crf 30 testt.mp4'
    ff_write_file = ff_write_file.format(w, h).split(' ')
    ff_read_file = 'ffmpeg -y -loglevel panic -i testt.mp4 -f image2pipe -pix_fmt gray -c:v rawvideo -'.split(' ')

    video_frame = np.zeros((h, w), dtype=np.uint8)
    training_frame = np.zeros((samples_per_frame, 64), dtype=np.float)

    autoencoder = Autoencoder(width, noise_std)
    #optimizer = optim.Adam(autoencoder.parameters(),lr = lr)
    optimizer = AdamW(autoencoder.parameters(),lr=lr,weight_decay=1e-3)

    if os.path.isfile('autoencoder.save'):
        try:
            save = torch.load('autoencoder.save')
            autoencoder.load_state_dict(save['model'])
            optimizer.load_state_dict(save['optim'])
            d = {'enc':autoencoder.encoder.state_dict(),
                 'dec': autoencoder.decoder.state_dict()}
            torch.save(d,'coder.save')

        except:
            autoencoder = Autoencoder(width)
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            print('Generating now NN')
            sleep(0.5)


    for param_group in optimizer.param_groups:
        a = param_group['lr']
        if a >= lr:
            param_group['lr'] = lr


    if learn:
        torch.autograd.set_detect_anomaly(True)

        frame_iter = iter(frame_samples_gen(100000,renew=True))

        criterion = nn.BCELoss()
        frame_back_samples = np.zeros((samples_per_frame,width),dtype=np.float)
        frame_back_reenc = np.zeros((samples_per_frame,64),dtype=np.float)

        i = 0
        while True:
            #frame_front_samples = next(frame_iter)
            frame_front_samples = np.random.randint(0,255,size=bytes_per_frame,dtype=np.uint8)
            frame_front_samples = coder.encode(frame_front_samples)
            i += 1

            frame_front_samples_t = torch.FloatTensor(frame_front_samples)

            optimizer.zero_grad()

            x, enc = autoencoder([frame_front_samples_t])
            x = x.view(-1,width)

            loss = criterion(x,frame_front_samples_t)

            loss.backward()
            optimizer.step()
            #with torch.no_grad():
            frame_back_samples_t = frame_front_samples_t.clone()
            acc = binary_accuracy(x, frame_front_samples_t)
            print(loss.item(), acc.item())

            save_dct = {'model': autoencoder.state_dict(),
                        'optim': optimizer.state_dict()}
            if i%2==0:
                if os.path.isfile('autoencoder.save'):
                    torch.save(save_dct, 'autoencoder.save.bak')
                    if os.path.exists('autoencoder.save'):
                        os.remove('autoencoder.save')
                    os.rename('autoencoder.save.bak','autoencoder.save')
                else:
                    torch.save(save_dct, 'autoencoder.save')

    else:

        #do testing
        frame_cnt = frame_cnt

        if not youtube_testing:
            codec = ffmpeg(ff_write_file,use_stdin=True)
            codec.start()
            sleep(2)
            i = 0

            for frame_samples in frame_samples_gen(frame_cnt):
                i += 1

                #a = coder.encode(coder.decode(frame_samples))
                a = frame_samples
                a = torch.Tensor(a)
                with torch.no_grad():
                    pred = autoencoder.encoder([a])
                    back_samples = torch.Tensor(frame_samples)
                    pred = pred * 255
                    pred = pred.view(-1).numpy().astype(np.uint8)

                #frame_from_samples(video_frame, pred)
                #codec.write(video_frame)
                codec.write(pred)
                print(i)
            codec.write_eof()
            while codec.is_running():
                sleep(0.1)

            print('write ended')

        codec = ffmpeg(ff_read_file,use_stdout=True)
        codec.start()
        sleep(1)

        hist = np.zeros((10000),dtype=np.uint64)
        no_error_range = 0

        back_frame = torch.zeros((samples_per_frame,64),dtype=torch.float)
        cnt = 0

        i = 0
        for frame_samples in frame_samples_gen(frame_cnt-1):
            i+=1
            frame = codec.readout(h * w).reshape(h, w)
            #frame_to_samples(frame, training_frame)

            with torch.no_grad():
                #x = torch.FloatTensor(training_frame)
                frame = frame.astype(np.float).reshape(-1)/255
                x = torch.FloatTensor(frame)

                #frame_diff = x - back_frame
                pred = autoencoder.decoder([x])
                back_frame = x

                pred.detach_()
                pred = pred.numpy()

                x = torch.FloatTensor(frame_samples)

                front_frame_enc = autoencoder.encoder([x])

                #front_frame_enc = front_frame_enc.detach().numpy()
                front_frame_enc = front_frame_enc.detach().view(-1).numpy()
                back_samples = torch.Tensor(frame_samples)



            pixels_enc = front_frame_enc.astype(np.float)

            test_data = coder.decode(pred).astype(np.int)

            truth_data = coder.decode(frame_samples).astype(np.int)



            b = a = np.equal(truth_data,test_data)
            a = a.astype(np.float)
            for x in np.nditer(b):
                cnt += 1
                if x:
                    no_error_range += 1
                else:
                    if no_error_range >= 10000:
                        no_error_range = 10000-1
                    hist[no_error_range] = hist[no_error_range] + 1
                    no_error_range = 0
            #dif = (pixels_enc-training_frame)
            dif = (pixels_enc - frame)

            print('{:.6f}'.format((np.mean(a))),'{:.6f}'.format(np.mean(dif)),'{:.6f}'.format(np.std(dif)))
            print(i)
        hist = hist / cnt
        import  matplotlib.pyplot as plt
        plt.plot(hist)
        plt.show()














        
