import numpy as np

import torch
from torch import nn
import torch.nn.functional as F



import argparse
import os,sys
from sys import exit
import pickle
from time import sleep
import random

from ffmpegwrapper import ffmpeg

#try:
import creedsolo as reedsolo



w = 1920
h = 1080
bits_per_input = 8*4
bytes_per_input = bits_per_input // 8
samples_per_frame = w*h//64
bytes_per_frame = samples_per_frame*bytes_per_input
#features_mult = 20
ff_write_file = 'ffmpeg -y -loglevel panic -f rawvideo -pix_fmt gray -s:v {0}x{1} -r 60 -i - -c:v libx264 -preset veryfast -crf 25 ?filename?'
ff_write_file = ff_write_file.format(w, h).split(' ')
ff_read_file = 'ffmpeg -y -loglevel panic -i ?filename? -f image2pipe -pix_fmt gray -c:v rawvideo -'.split(' ')
from binary_encoder import BinaryCoder
coder = BinaryCoder(bits_per_input)
width = coder.get_width()
f1 = width*2
f2 = width//2
f3 = width//4




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
    parser.add_argument('--output_file','-o', default=None, help='encoding output file name (MP4)')
    parser.add_argument('-e', action='store_const', const=True, default=False, help='perform encoding phase')
    parser.add_argument('-d', action='store_const', const=True, default=False, help='perform decoding phase')
    parser.add_argument('-t', default=None, help='optional path to file to store inbetween data and do some testing on '+
                        'decoding phase')
    group = parser.add_argument_group(description='Params related to error correction')
    group.add_argument('-fecdatasize', nargs='?', const=True, default=40, type=int, help='Size of data block')
    group.add_argument('-feccodes', nargs='?', const=True, default=5, type=int, help='Number of errors to be currected '
                        +'per full block')

    args = parser.parse_args()
    #args = parser.parse_args(['test.dat','-o','testt2.mp4','-e','-t','cache.dat'])
    #args = parser.parse_args(['testt2.mp4','-d','-t','cache.dat'])




    '''
    if args.t is not None:
        assert args.output_file is not None
        f = open('test.dat','wb')
        #i = list(range(args.fecdatasize+5))
        #i = bytes(i)
        #_bpf = bytes_per_frame * args.fecdatasize // (args.fecdatasize + args.feccodes*2)
        ar = bytes([random.randint(0,255) for _ in range(1000)])
        for _ in range(12300):
            f.write(ar)
        f.close()
        del ar
    '''


    #path to NN weigths
    try:
        SAVE_FILE = sys._MEIPASS + '\\coder.dat'
    except:
        SAVE_FILE = os.path.dirname(os.path.realpath(__file__)) + '\\coder.dat'


    #SAVE_FILE = r"D:\FILES\Code\20190419 Videofy python 3\1\dist\programm\main\coder.dat"


    random.seed(0)
    rand_masks = [None, None] #xor masks on source data to make
    rand_masks[0] = [random.randint(0,255) for i in range(bytes_per_frame)]
    rand_masks[0] = np.array(rand_masks[0], dtype=np.uint8)
    rand_masks[1] = [random.randint(0,255) for i in range(bytes_per_frame)]
    rand_masks[1] =  np.array(rand_masks[1], dtype=np.uint8)

    do_check = args.t is not None

    fk = args.fecdatasize
    fn = args.feccodes * 2 + fk


    if args.e:
        assert args.output_file is not None
        assert args.output_file.split('.')[-1].lower() == 'mp4'
        ff_write_file[-1] = args.output_file
        f = open(args.input_file, 'rb')
        bytes_todo = os.path.getsize(args.input_file)

        fk = args.fecdatasize
        fn = args.feccodes*2 + fk
        path = args.input_file
        header = header_make(path)

        file_read_size = bytes_per_frame * fk // fn
        assert (bytes_per_frame * fk % fn) == 0
        #fsize = os.path.getsize(path)
        nnecoder = Encoder(width)
        #save = torch.load('coder.dat')

        save = torch.load(SAVE_FILE)
        nnecoder.load_state_dict(save['enc'])

        rs = reedsolo.RSCodec(nsym=fn-fk,nsize=fn)
        codec = ffmpeg(ff_write_file,use_stdin=True)
        codec.start()
        sleep(2)

        f = open(path, 'rb')
        if do_check:
            f_check = open(args.t,'wb')
        #random.seed(0)
        #data = bytes([random.randint(0,255) for _ in range(file_read_size)])
        data = np.random.randint(0,255,file_read_size,dtype=np.uint8)
        data = np.frombuffer(data,dtype=np.uint8)
        #data = np.random.randint(0, 255, bytes_per_frame, dtype=np.uint8)

        data[:len(header)] = np.frombuffer(header,dtype=np.uint8)
        #data = bytearray(data.tobytes())
        video_frame = np.empty((h,w),dtype=np.uint8)
        i = 0
        while True:
            i += 1

            data = data.tobytes()
            data = rs.encode(data)
            if do_check:
                f_check.write(data)
            data = np.frombuffer(data,dtype=np.uint8)

            data = np.bitwise_xor(data, rand_masks[i%2])
            #rand_mask = rand_mask*7

            data = coder.encode(data)
            with torch.no_grad():
                data = torch.FloatTensor(data)
                data = nnecoder([data])
                data = data.numpy() * 255
                #data = data.detach().numpy().astype(np.uint8).reshape(h*w)
            data = data.astype(np.uint8).reshape(h*w)
            #frame_from_samples(video_frame, data)
            codec.write(data)

            data = f.read(file_read_size)
            #data = f.read(bytes_per_frame)
            if len(data)==0:
                break
            if len(data) < file_read_size:
                data = data + np.random.randint(0, 255, (file_read_size - len(data)), dtype=np.uint8).tobytes()[:]
            #if len(data) < bytes_per_frame:
            #    data = data + np.random.randint(0,255,(bytes_per_frame-len(data)),dtype=np.uint8).tobytes()[:]
               # l = len(data)
            data = np.frombuffer(data,dtype=np.uint8)
            bytes_todo -= len(data)
            if bytes_todo < 0:
                bytes_todo = 0
            print("Frame #{0} Data remaining: {1}".format(i,bytes_todo))
        codec.write_eof()
        if do_check:
            f_check.close()
        while codec.is_running():
            sleep(0.1)
        exit(0)

    if args.d:
        ff_read_file[5] = args.input_file
        fk = args.fecdatasize
        fn = args.feccodes * 2 + fk
        path = args.input_file

        file_read_size = bytes_per_frame * fk // fn
        assert (bytes_per_frame * fk % fn) == 0
        # fsize = os.path.getsize(path)
        nndecoder = Decoder(width)
        #save = torch.load('coder.dat')
        save = torch.load(SAVE_FILE)
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

        bytes_todo = None

        i=0
        while True:
            i+=1

            #frame = codec.readout(h*w).reshape((h,w))
            frame = codec.readout(h * w)
            #frame = frame.astype(np.float).reshape(-1) / 255

            #frame_to_samples(frame,samples_frame)
            with torch.no_grad():
                #data = torch.tensor(samples_frame,dtype=torch.float)
                data = torch.FloatTensor(frame)/255
                data = nndecoder([data])
                data = data.numpy().reshape((-1,width))
            data = coder.decode(data)
            data = np.bitwise_xor(data, rand_masks[i%2])
            #rand_mask = rand_mask * 7
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
                bytes_todo = size = d['size']

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
            bytes_todo -= len(data)
            if bytes_todo < 0:
                bytes_todo = 0
            print("Frame #{0} Data remaining: {1}".format(i, bytes_todo))
            size -= len(data)
        exit(0)

    if (args.t is not None) and False: #turned off for some time
        fk = args.fecdatasize
        fn = args.feccodes * 2 + fk
        file_read_size = bytes_per_frame * fk // fn

        nnecoder = Encoder(width)
        nndecoder = Decoder(width)
        save = torch.load('coder.save')
        nnecoder.load_state_dict(save['enc'])
        nndecoder.load_state_dict(save['dec'])

        codec = ffmpeg(ff_write_file,use_stdin=True)
        codec.start()
        sleep(2)

        rs = reedsolo.RSCodec(nsym=fn - fk, nsize=fn)
        std = 50.0/255
        i = 0
        frame_count = 20
        random.seed(0)
        for i in range(frame_count):
            rand = bytes([random.randint(0,255) for _ in range(file_read_size)])
            rand = np.frombuffer(rand,dtype=np.uint8)
            dns = rand
            drss = rs.encode(dns.tobytes())
            drsns = np.frombuffer(drss,dtype=np.uint8)
            dbs = coder.encode(drsns)
            #x = databincodersrc.reshape((-1,width))
            with torch.no_grad():
                dts = torch.tensor(dbs,dtype=torch.float32)
                dnns = nnecoder([dts])
                dnnns = dnns.numpy()*255
            dnnns = dnnns.astype(np.uint8).reshape(h * w)
            codec.write(dnnns)
            print('write',i)
        codec.write_eof()
        while codec.is_running():
            sleep(0.1)



        # =========================================================



        codec = ffmpeg(ff_read_file,use_stdout=True)
        codec.start()
        sleep(2)
        random.seed(0)
        for i in range(frame_count):
            rand = bytes([random.randint(0, 255) for _ in range(file_read_size)])
            rand = np.frombuffer(rand, dtype=np.uint8)
            dns = rand
            drss = rs.encode(dns.tobytes())
            drsns = np.frombuffer(drss, dtype=np.uint8)
            dbs = coder.encode(drsns)
            # x = databincodersrc.reshape((-1,width))
            with torch.no_grad():
                dts = torch.FloatTensor(dbs)
                dnns = nnecoder([dts])
                dnnns = dnns.numpy() * 255
            dnnns = dnnns.astype(np.uint8).reshape(h*w)

            dnnnd = codec.readout(h*w)

            x = dnnns.astype(float) - dnnnd.astype(float)
            x = np.abs(x)/255
            x = np.mean(x)
            print(i,'vframe diff ',x)

            with torch.no_grad():
                dnnd = torch.FloatTensor(dnnnd)/255
                dtd = nndecoder([dnnd])
                dbd = dtd.numpy().reshape((-1,width))

            x = np.abs(dbd-dbs)
            x = np.mean(x)
            print(i,'bin enc diff ',x)

            drsnd = coder.decode(dbd)

            x = np.mean(np.not_equal(drsnd,drsns))
            print(i,'mean err ',x)
            a = np.not_equal(drsnd,drsns)
            a = a.reshape((-1,fn))
            a = np.sum(a,-1)
            a = np.max(a)
            print(i,'max errs',a)








































