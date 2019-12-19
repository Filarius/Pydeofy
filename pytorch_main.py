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

from ffmpegwrapper import ffmpeg
from binary_encoder import BinaryCoder

from collections import namedtuple
DataMem = namedtuple('DataMem',[])

m = 6
w = 16*5*m
h = 16*5*m

#w = 1920
#h = 1080

h_t = 864 # 1080*0.8
fps = 60
seconds = 60
batch = 1
bits_per_input = 8*4
samples_per_frame = w*h//64

frame_cnt  = 10
batch_size = int(samples_per_frame*1)
batch_cnt = samples_per_frame//batch_size
epochs = 1
lr = 0.005
sleep_time = 0.4
noise_std = 0/255
features_mult = 20
learn = '0000'=='0000'
youtube_testing = '0000'=='000'
youtube_tuning = '0000'=='000'






coder = BinaryCoder(bits_per_input)
width = coder.get_width()

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
    frame_now = np.empty((samples_per_frame, width), dtype=np.uint8)
    for k in range(cnt):
        for i in range(samples_per_frame):
            frame_now[i, :] = encs[ind,:]
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
        self.ct1 = nn.ConvTranspose2d(in_channels=width, out_channels=2*features_mult, kernel_size=3)
        self.c2 = nn.Conv2d(in_channels=1, out_channels=features_mult, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=2*features_mult)
        self.bn2 = nn.BatchNorm2d(num_features=features_mult)
        self.fc1 = nn.Linear(2*features_mult * 3*3 + features_mult * 6*6, 64)

    def forward(self,inp):
        #inp = [data, diff_prev_pixels_pred]
        x = inp[0].view(-1,self._width,1,1)
        x = self.ct1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)  # flatten

        y = inp[1].view(-1, 1, 8, 8)
        y = self.c2(y)
        y = self.bn2(y)
        y = F.leaky_relu(y)
        y = torch.flatten(y, 1)

        x = torch.cat((x,y),dim=1)

        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = torch.clamp(x,min=0.0,max=1.0)
        return x



class Decoder(nn.Module):
    def __init__(self, width):
        super(Decoder, self).__init__()
        self.c2 = nn.Conv2d(in_channels=2, out_channels=features_mult, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=features_mult)
        self.fc2 = nn.Linear(features_mult * 6*6, width)

    def forward(self, inp):
        #inp = [now_pixels, diff_before_pixels]
        x = inp[0].view(-1,1,8,8)
        y = inp[1].view(-1,1,8,8)
        x = torch.cat([x,y],dim=1)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x,1)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


class Autoencoder(nn.Module):
    def __init__(self, width, noise_std=None):
        super(Autoencoder,self).__init__()
        self._noise_std = noise_std
        self._width = width
        self.decoder = Decoder(width)
        self.encoder = Encoder(width)

    def forward(self,inp):
        # data, diff_truth_pixels, diff_prev_pixels_predict, diff_prev_pixels
        x = self.encoder([inp[0], inp[2]])
        x = x+torch.Tensor(inp[1])
        if self._noise_std is not None:
            x = x + self._noise_std*torch.randn_like(x)
        x = x.clamp(0.0,1.0)
        x = self.decoder([x, inp[3]])
        return x


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

    ff_write_file = 'ffmpeg -y -loglevel panic -f rawvideo -pix_fmt gray -s:v {0}x{1} -r 60 -i - -c:v libx264 -preset fast -crf 25 testt.mp4'
    ff_write_file = ff_write_file.format(w, h).split(' ')
    ff_read_file = 'ffmpeg -y -loglevel panic -i testt.mp4 -f image2pipe -pix_fmt gray -c:v rawvideo -'.split(' ')

    video_frame = np.zeros((h, w), dtype=np.uint8)
    training_frame = np.zeros((samples_per_frame, 64), dtype=np.float)



    autoencoder = Autoencoder(width)
    #optimizer = optim.Adam(autoencoder.parameters(),lr = lr)
    optimizer = AdamW(autoencoder.parameters(),lr=lr)

    if os.path.isfile('autoencoder.save'):
        try:
            save = torch.load('autoencoder.save')
            autoencoder.load_state_dict(save['model'])
            optimizer.load_state_dict(save['optim'])
        except:
            autoencoder = Autoencoder(width)
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    for param_group in optimizer.param_groups:
        a = param_group['lr']
        if a >= lr:
            param_group['lr'] = lr


    if learn:
        ffmpeg_h264 = ffmpeg(ff_write,use_stdin=True,use_stdout=True)
        ffmpeg_gray = ffmpeg(ff_read,use_stdin=True,use_stdout=True)
        ffmpeg_h264.start()
        ffmpeg_gray._process = Popen(ff_read, stdin=ffmpeg_h264._process.stdout, stdout=-1) # connect both ffmpeg in 1 pipe line
        #no need to start ffmpeg_gray
        sleep(1)
        qgray = Queue()
        fgray = Thread(target=pipe_to_queue, args=(ffmpeg_gray, qgray, h * w))
        fgray.daemon = True
        fgray.start()
        print("PAUSE!")
        sleep(1)
        frames_queue = Queue()
        skip_cnt = 0
        frame_id_wait = 1
        blank = np.zeros((samples_per_frame, 64), dtype=np.float)
        lr_reducer = lr
        lr_counter = 0
        best_loss = 1
        frame_cnt = 2000000
        frame_iter = iter(frame_samples_gen(frame_cnt,renew=True))
        i = 0
        frame_buf_size = 0
        pixels_back_t = torch.FloatTensor(np.zeros_like(training_frame))
        pixels_front_t = torch.FloatTensor(np.zeros_like(training_frame))
        pixels_lst = [pixels_front_t, pixels_back_t]

        frame_prev_enc = np.zeros_like(training_frame)
        while True:
            frame_samples = next(frame_iter)
            i += 1
            with torch.no_grad():
                frame_samples_t = torch.FloatTensor(frame_samples)

                frame_pred_t = autoencoder.encoder([frame_samples_t, pixels_back_t])
                pixels_back_t = frame_pred_t

                frame_pred = frame_pred_t.numpy()

            frame_from_samples(video_frame, frame_pred)
            x = np.copy(frame_samples)
            frames_queue.put((i,
                              frame_pred_t,
                              torch.FloatTensor(x),
                              pixels_back_t))
            ffmpeg_h264.write(video_frame)
            frame_buf_size += 1

            mean = np.mean(video_frame)
            std = np.std(video_frame)

            while not qgray.empty():
                frame = qgray.get()
                k, pixels, data, pixels_enc= frames_queue.get()
                frame_buf_size -= 1
                if k != frame_id_wait:
                    frame_prev_enc = frame
                    continue
                frame_id_wait = k + frame_buf_size+2

                frame = frame.reshape((h, w))
                frame_prev_enc = frame_prev_enc.reshape((h,w))
                frame_to_samples(frame, training_frame)
                frame1 = np.copy(training_frame)
                frame_to_samples(frame_prev_enc, training_frame)
                frame2 = np.copy(training_frame)
                frame_prev_enc = np.copy(frame)


                frame = np.copy(training_frame)

                training_frame_t = torch.FloatTensor(training_frame)
                diff = (training_frame_t - pixels)
                diff = 1 * diff
                metric = diff.abs().mean().item()
                a = pixels.shape[0]
                b = a - batch_size
                criterion = nn.BCELoss()

                prev_pixels_diff = frame1 - frame2

                for batch_i in range(1):
                    ind_l = random.randint(0, b)
                    ind_r = ind_l + batch_size
                    optimizer.zero_grad()
                    x = data[ind_l:ind_r]
                    x = torch.Tensor(x)
                    x_diff = torch.FloatTensor(diff[ind_l:ind_r])
                    x_diff_prev_pixels_pred = pixels_enc[ind_l:ind_r]
                    x_diff_prev_pixels = torch.FloatTensor(prev_pixels_diff[ind_l:ind_r])

                    y = autoencoder([x,
                                     x_diff,
                                     x_diff_prev_pixels_pred,
                                     x_diff_prev_pixels])
                    loss = criterion(y,x)
                    loss.backward()
                    optimizer.step()
                    acc = binary_accuracy(y,x)
                    print(loss.item(),acc.item(),'>', mean, std)



                save_dct = {'model':autoencoder.state_dict(),
                            'optim':optimizer.state_dict()}
                torch.save(save_dct,'autoencoder.save')

                for param_group in optimizer.param_groups:
                    a = param_group['lr']
                    if a >= lr:
                        param_group['lr'] = lr



    else:

        #do testing
        frame_cnt = frame_cnt
        if not youtube_testing:
            codec = ffmpeg(ff_write_file,use_stdin=True)
            codec.start()
            sleep(2)
            i = 0

            back_frame = torch.zeros((samples_per_frame,64),dtype=torch.float32)
            for frame_samples in frame_samples_gen(frame_cnt):
                i += 1
                #a = coder.encode(coder.decode(frame_samples))
                a = frame_samples
                a = torch.Tensor(a)
                with torch.no_grad():
                    pred = autoencoder.encoder([a, back_frame])
                    back_frame = pred
                    pred = pred.numpy()

                frame_from_samples(video_frame, pred)
                codec.write(video_frame)
                print(i)
            codec.write_eof()
            while codec.is_running():
                sleep(0.1)

            print('write ended')

        codec = ffmpeg(ff_read_file,use_stdout=True)
        codec.start()
        sleep(1)

        if youtube_tuning:
            print('TUNING')
            i=0
            back_frame = torch.zeros((samples_per_frame,64),dtype=torch.float32)
            back_frame_diff = torch.zeros_like(back_frame)
            back_frame_enc = torch.zeros_like(back_frame)
            back_frame_enc_diff = torch.zeros_like(back_frame)
            for frame_samples in frame_samples_gen(frame_cnt):
                i += 1
                frame = codec.readout(h * w).reshape(h, w)
                with torch.no_grad():
                    samples = torch.FloatTensor(frame_samples)

                    frame_pred = autoencoder.encoder([samples, back_frame_diff])
                    frame_pred.detach_()
                    back_frame_diff = frame_pred - back_frame
                    back_frame = frame_pred
                    frame_pred = frame_pred.numpy()
                frame_to_samples(frame, training_frame)

                diff = training_frame - frame_pred
                back_frame_enc_diff = torch.Tensor(training_frame) - back_frame_enc
                back_frame_enc = torch.Tensor(training_frame)
                criterion1 = nn.BCELoss()
                for batch_i in range(batch_cnt):
                    ind_l = batch_size*batch_i
                    ind_r = ind_l + batch_size
                    optimizer.zero_grad()
                    x = frame_samples[ind_l:ind_r]
                    x = torch.Tensor(x)
                    x_diff = torch.Tensor(diff[ind_l:ind_r])
                    x_back_frame_diff = back_frame_diff[ind_l:ind_r]
                    x_back_frame_enc_diff = back_frame_enc_diff[ind_l:ind_r]
                    y = autoencoder([x,
                                     x_diff,
                                     x_back_frame_diff,
                                     x_back_frame_enc_diff])
                    loss = criterion1(y, x)
                    loss.backward()
                    optimizer.step()
                    acc = binary_accuracy(y, x)
                    print(i, batch_i, loss.item(), acc.item())
                save_dct = {'model': autoencoder.state_dict(),
                            'optim': optimizer.state_dict()}
                torch.save(save_dct, 'autoencoder.save')

                for param_group in optimizer.param_groups:
                    a = param_group['lr']
                    if a >= lr:
                        param_group['lr'] = lr
            exit(0)



        i = 0
        back_frame = torch.zeros((samples_per_frame, 64), dtype=torch.float32)
        front_frame = torch.zeros_like(back_frame)
        back_frame_diff = torch.zeros_like(back_frame)
        back_frame_enc = torch.zeros_like(back_frame)
        back_frame_enc_diff = torch.zeros_like(back_frame)
        for frame_samples in frame_samples_gen(frame_cnt):
            i+=1
            frame = codec.readout(h * w).reshape(h, w)
            frame_to_samples(frame, training_frame)
            with torch.no_grad():
                x = torch.FloatTensor(training_frame)
                back_frame_diff = x - back_frame
                pred = autoencoder.decoder([x, back_frame_enc_diff])
                back_frame = x

                pred.detach_()
                pred = pred.numpy()

                x = torch.FloatTensor(frame_samples)

                front_frame_enc = autoencoder.encoder([x,back_frame_diff])
                back_frame_enc_diff = front_frame - back_frame
                back_frame = front_frame

                front_frame_enc = front_frame.detach_().numpy()

            pixels_enc = front_frame_enc.astype(np.float)

            test_data = coder.decode(pred).astype(np.int)

            truth_data = coder.decode(frame_samples).astype(np.int)



            a = np.equal(truth_data,test_data)
            a = a.astype(np.float)

            dif = (pixels_enc-training_frame)

            print('{:.6f}'.format((np.mean(a))),'{:.6f}'.format(np.mean(dif)),'{:.6f}'.format(np.std(dif)))













        
