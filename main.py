import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import zfec

import argparse
import os
import pickle
from time import sleep
import zfec
from io import BytesIO
import random

from ffmpegwrapper import ffmpeg

try:
    import сreedsolo as reedsolo
except:
    import reedsolo



w = 1920
h = 1080
bits_per_input = 8*4
bytes_per_input = bits_per_input // 8
samples_per_frame = w*h//64
bytes_per_frame = samples_per_frame*bits_per_input//8
features_mult = 20
ff_write_file = 'ffmpeg -y -loglevel panic -f rawvideo -pix_fmt gray -s:v {0}x{1} -r 60 -i - -c:v libx264 -preset fast -crf 25 testt.mp4'
ff_write_file = ff_write_file.format(w, h).split(' ')
ff_read_file = 'ffmpeg -y -loglevel panic -i testt.mp4 -f image2pipe -pix_fmt gray -c:v rawvideo -'.split(' ')


from binary_encoder import BinaryCoder
coder = BinaryCoder(bits_per_input)
width = coder.get_width()

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

class Encoder(nn.Module):
    def __init__(self, width):
        super(Encoder,self).__init__()
        self._width = width
        self.ct1 = nn.ConvTranspose2d(in_channels=width, out_channels=features_mult, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=features_mult)
        self.fc1 = nn.Linear(features_mult * 9, 64)

    def forward(self,inp):
        x = inp[0].view(-1,self._width,1,1)
        x = self.ct1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = torch.flatten(x, 1)
        #x = nn.Dropout(0.1)(x)

        x = self.fc1(x)
        x = torch.sigmoid(x)
        x= x.clamp(min=0.0,max=1.0)
        return x


class Decoder(nn.Module):
    def __init__(self, width):
        super(Decoder, self).__init__()
        self.c2 = nn.Conv2d(in_channels=1, out_channels=features_mult, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=features_mult)
        self.fc2 = nn.Linear(features_mult * 6*6, width)

    def forward(self, inp):
        x = inp[0].view(-1,1,8,8)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x,1)
        #x = nn.Dropout(0.1)(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def binary_accuracy(check, target):
    '''Binary accuracy metric'''
    check = check >= 0.5
    target = target >= 0.5
    ret = check.eq(target)
    ret = ret.sum().type(torch.float32)
    ret = ret / target.numel()
    return ret

def file_read_generator(path,chunk_size):
    f = open(path)
    while True:
        ret = f.read(chunk_size)
        if len(ret)==0:
            break
        yield ret

def header_make(path):
    """
    :param path:
        path - path to file
    :return:
        pickled info header
    """
    filename = os.path.basename(path)
    filesize = os.path.getsize(path)
    ret = {'name':filename,
           'size':filesize}
    ret = pickle.dumps(ret)
    return ret

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Simple app whats use MP4 as data container. '+
                                     'At encoding phase input file can be any file with data. '+
                                     'At decoding phase input must be a video file based on one from encoding phase\n'+
                                     'Use testing to check errors count.')
    parser.add_argument('input_file',help='path to input file')
    parser.add_argument('-e', action='store_const', const=True, default=False, help='perform encoding phase')
    parser.add_argument('-d', action='store_const', const=True, default=False, help='perform decoding phase')
    parser.add_argument('-t', default=None, help='optional path to file to store inbetween data and do some testing on '+
                        'decoding phase')
    group = parser.add_argument_group(description='Params related to error correction')
    group.add_argument('-fecdatasize', nargs='?', const=True, default=44, type=int, help='Size of data block')
    group.add_argument('-feccodes', nargs='?', const=True, default=3, type=int, help='Number of errors to be currected '
                        +'per full block')

    args = parser.parse_args(['test.dat','-e','-t','cache.dat'])
    #
    args = parser.parse_args(['testt.mp4', '-d','-t','cache.dat'])
    #args = parser.parse_args(['testt.mp4', '-d'])
    #args = parser.parse_args(['-h'])


    f = open('test.dat','wb')
    i = list(range(args.fecdatasize+5))
    i = bytes(i)
    #_bpf = bytes_per_frame * args.fecdatasize // (args.fecdatasize + args.feccodes*2)
    ar = bytes([x for x in range(97)])
    for _ in range(230*108*5):
        f.write(ar)
    f.close()
    del ar

    random.seed(0)
    rand_mask = [random.randint(0,255) for i in range(bytes_per_frame)]
    rand_mask = np.array(rand_mask,dtype=np.uint8) #xor mask on source data

    do_check = args.t is not None

    if args.e:
        fk = args.fecdatasize
        fn = args.feccodes*2 + fk
        path = args.input_file
        header = header_make(path)

        file_read_size = bytes_per_frame * fk // fn
        assert (bytes_per_frame * fk % fn) == 0
        #fsize = os.path.getsize(path)
        nnecoder = Encoder(width)
        save = torch.load('coder.save')
        nnecoder.load_state_dict(save['enc'])

        rs = reedsolo.RSCodec(nsym=fn-fk,nsize=fn)
        codec = ffmpeg(ff_write_file,use_stdin=True)
        codec.start()
        sleep(2)

        f = open(path, 'rb')
        if do_check:
            f_check = open(args.t,'wb')
        data = np.random.randint(0,255,file_read_size,dtype=np.uint8)
        #data = np.random.randint(0, 255, bytes_per_frame, dtype=np.uint8)

        data[:len(header)] = np.frombuffer(header,dtype=np.uint8)
        #data = bytearray(data.tobytes())
        video_frame = np.empty((h,w),dtype=np.uint8)
        i = 0
        while True:
            i += 1
            print(i)
            data = data.tobytes()
            data = rs.encode(data)
            if do_check:
                f_check.write(data)
            data = np.frombuffer(data,dtype=np.uint8)

            data = np.bitwise_xor(data, rand_mask)
            rand_mask = rand_mask*7

            data = coder.encode(data)
            with torch.no_grad():
                data = torch.tensor(data,dtype=torch.float)
                data = nnecoder([data]).detach().numpy()
            frame_from_samples(video_frame, data)
            codec.write(video_frame)

            data = f.read(file_read_size)
            #data = f.read(bytes_per_frame)
            if len(data)==0:
                break
            if len(data) < file_read_size:
                data = data + np.random.randint(0, 255, (file_read_size - len(data)), dtype=np.uint8).tobytes()[:]
            #if len(data) < bytes_per_frame:
            #    data = data + np.random.randint(0,255,(bytes_per_frame-len(data)),dtype=np.uint8).tobytes()[:]
                l = len(data)

            data = np.frombuffer(data,dtype=np.uint8)
        codec.write_eof()
        if do_check:
            f_check.close()
        while codec.is_running():
            sleep(0.1)
        exit(0)

    if args.d:
        fk = args.fecdatasize
        fn = args.feccodes * 2 + fk
        path = args.input_file

        file_read_size = bytes_per_frame * fk // fn
        assert (bytes_per_frame * fk % fn) == 0
        # fsize = os.path.getsize(path)
        nndecoder = Decoder(width)
        save = torch.load('coder.save')
        nndecoder.load_state_dict(save['dec'])

        rs = reedsolo.RSCodec(nsym=fn - fk, nsize=fn)
        codec = ffmpeg(ff_read_file, use_stdout=True)
        codec.start()
        sleep(2)

        f = None
        if do_check:
            f_check = open(args.t,'rb')
        size = 0

        samples_frame = np.empty((samples_per_frame, 64), dtype=np.float)

        i=0
        while True:
            i+=1
            print(i)
            frame = codec.readout(h*w).reshape((h,w))

            frame_to_samples(frame,samples_frame)
            with torch.no_grad():
                data = torch.tensor(samples_frame,dtype=torch.float)
                data = nndecoder([data])
                data = data.detach().numpy()
            data = coder.decode(data)
            data = np.bitwise_xor(data, rand_mask)
            rand_mask = rand_mask * 7
            data = data.tobytes()

            if do_check:
                d = data
                x = f_check.read(len(d))
                y = np.frombuffer(d,dtype=np.uint8)
                x = np.frombuffer(x, dtype=np.uint8)
                b = np.not_equal(x,y).astype(np.int).astype(np.float)

                er_mean = np.mean(b)
                b = b.reshape(-1, fn)
                b = np.sum(b,axis=-1)
                er_dense = np.max(b)
                print('err mean {:.07f} max err per block {}'.format(er_mean,er_dense))


            data = rs.decode(data)
            if i==1:
                d = pickle.loads(data)
                fname = d['name']
                size = d['size']
                if os.path.exists(fname):
                    name,ext = fname.split('.')
                    newname = name
                    k = 0
                    while os.path.exists('.'.join([newname,ext])):
                        newname = name + '({0})'.format(k)
                        k += 1
                    fname = '.'.join([newname,ext])

                f = open(fname,'wb')
                continue
            if size > len(data):
                f.write(data)
            else:
                f.write(data[0:size])
                f.close()
                break
            size -= len(data)
        exit(0)

    if args.t:
        fk = args.fecdatasize
        fn = args.feccodes * 2 + fk
        file_read_size = bytes_per_frame * fk // fn

        nnecoder = Encoder(width)
        nndecoder = Decoder(width)
        save = torch.load('coder.save')
        nnecoder.load_state_dict(save['enc'])
        nndecoder.load_state_dict(save['dec'])

        rs = reedsolo.RSCodec(nsym=fn - fk, nsize=fn)
        std =0.0
        i = 48
        while True:
            datanumpysrc = np.random.randint(0,255,file_read_size,dtype=np.uint8)
            datarssrc = rs.encode(datanumpysrc.tobytes())
            datarsnpsrc = np.frombuffer(datarssrc,dtype=np.uint8)
            databincodersrc = coder.encode(datarsnpsrc)
            with torch.no_grad():
                datatensorsrc = torch.tensor(databincodersrc,dtype=torch.float32)
                datannsrc = nnecoder([datatensorsrc]).detach()
                datannsrc = datannsrc + torch.randn_like(datannsrc)*std
                datatensordst = nndecoder([datannsrc]).detach()
                databincoderdst = datatensordst.numpy()
            datarsnpdst = coder.decode(databincoderdst)
            x = np.not_equal(datarsnpsrc,datarsnpdst)
            a = np.sum(x)
            b = np.mean(x)
            x = x.reshape((-1,100))
            x = np.sum(x,axis=-1)
            x = np.max(x)
            print(i,x,str(std)[:5],a,str(b)[:10])
            std = i/255
            #i+=1

            datarsdst = datarsnpdst.tobytes()
            datanumpydst = np.frombuffer(rs.decode(datarsdst),dtype=np.uint8)
            b = np.mean(np.not_equal(datanumpydst, datanumpysrc))



















