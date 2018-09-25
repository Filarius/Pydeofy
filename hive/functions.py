import numpy as np
import sys
import multiprocessing
import io
from time import sleep
from hive import fifo
from math import ceil
from scipy.fftpack import dctn,idctn
from reedsolo import RSCodec
from array import array
import itertools

coder = RSCodec(50)

s = bytearray(255)
s = coder.encode(s)
print(len(s))
a = 2

def bytes_to_rs(data):
    return  coder.encode(data.tobytes())

def bytes_from_rs(data):
    return np.array(coder.decode(data),dtype=np.uint8)


def bytes_to_bits(arr):
    arr = np.array(arr,dtype=np.uint8)
    a = arr
    return np.unpackbits(arr)

def  bits_to_bytes(arr):
    if (len(arr) % 8) != 0:
        raise AttributeError
    arr = np.array(arr)
    return np.packbits(arr)

def  bits_to_cell(arr,size,dtype=np.uint8):
    if (len(arr) % size) != 0:
        raise AttributeError
    if size > 8:
        raise AttributeError
    t = np.zeros(len(arr) // size,dtype=dtype)

    k = 0
    i = 0
    j = 0
    while i < len(arr):
        k = k * 2
        k = k + arr[i]
        i = i + 1
        if (i % size) == 0:
            t[j] = k
            j = j + 1
            k = 0
    return t

def cell_to_bits(arr,size):
    i = 0
    #t = bytearray(len(arr) * size)
    t = np.zeros(len(arr) * size,dtype=np.uint8)
    j = 0
    k = arr[i]
    s = size - 1
    q = k
    while True:
        t[j+s] = k % 2
        k = k // 2
        s = s - 1
        if s == -1:
            s = size - 1
            j = j + size
            i = i + 1
            if i == len(arr):
                break
            k = arr[i]
            q = k
    return t








a = [(0,0), (0,1) ,(1,0),
    (2,0),(1,1),(0,2)]

_snake_encode8 = [
    (0,0), (0,1) ,(1,0),
    (2,0),(1,1),(0,2),
(0,3),(1,2),(2,1),(3,0),
(4,0),(3,1),(2,2),(1,3),(0,4),
(0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
(6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),
(7,0),(6,1),(5,2),(4,3),(3,4),(2,5),(1,6),(0,7),
(1,7),(2,6),(3,5),(4,4),(5,3),(6,2),(7,1),
(7,2),(6,3),(5,4),(4,5),(3,6),(2,7),
(3,7),(4,6),(5,5),(6,4),(7,3),
(7,4),(6,5),(5,6),(4,7),
(5,7),(6,6),(7,5),
(7,6),(6,7),
(7,7)
]

_snake_encode8_plain = [x + y*8 for y,x in _snake_encode8]
_snake_encode8_plain_np = np.empty(64,dtype=np.uint8)
for i,x in enumerate(_snake_encode8_plain):
    _snake_encode8_plain_np[x] = i
_snake_decode8_plain_np = np.zeros(64,dtype=np.uint8)
for i,x in enumerate(_snake_encode8_plain_np):
    _snake_decode8_plain_np[x] = i

'''
a = np.arange(0,64)
b = a[_snake_encode8_plain_np]
c = b[_snake_decode8_plain_np]
d = 1
'''

_snake_encode4 = [(0,0),
(0,1),(1,0),
(2,0),(1,1),(0,2),
(0,3),(1,2),(2,1),(3,0),
(3,1),(2,2),(1,3),
(2,3),(3,2),
(3,3)
]

_snake_encode4_plain = [x + y*4 for x,y in _snake_encode4]
_snake_encode4_plain_np = np.array(_snake_encode4_plain)
#def YFrameTo



def snake_unpack8(arr):
#place values from chunks into 8x8 flat arrays with snake order
    assert (arr.shape[0] % 64) == 0

    temp = np.zeros(arr.shape,dtype=np.uint8)
    i=0
    while i< (arr.shape[0]-1):
        j = 0
        while j < 64:
            temp[i + j] = arr[i + _snake_encode8_plain[j]]
            j += 1
        i += 64
    return temp

def snake_pack8_np(data):
    assert (data.shape[0] == 64)
    result = np.empty(data.shape,dtype=np.int32)
    for i in range(len(data)//64):
        result[i*64:i*64+63] = data[i*64:i*64+64][_snake_encode8_plain_np]
    return result

def snake_unpack8_np(data):
    assert (data.shape[0] % 64) == 0
    result = np.empty(data.shape,dtype=np.int32)
    for i in range(len(data)//64):
        result[i*64:i*64+63] = data[i*64:i*64+64][_snake_decode8_plain_np]
    return result

def snake_pack8(arr):
    assert  (arr.shape[0] % 64) == 0
    temp = np.zeros(arr.shape, dtype=np.uint8)
    i = 0
    while i < arr.shape[0]:
        j = 0
        while j < 64:
            temp[i + _snake_encode8_plain[j]] = arr[i + j]
            j += 1
        i += 64
    return temp


def DCT4x4(d):
    tmp = np.zeros((16,), dtype=np.int32)
    result = np.zeros((16,), dtype=np.int32)
    for i in range(4):
        s03 = d[i * 4 + 0] + d[i * 4 + 3]
        s12 = d[i * 4 + 1] + d[i * 4 + 2]
        d03 = d[i * 4 + 0] - d[i * 4 + 3]
        d12 = d[i * 4 + 1] - d[i * 4 + 2]

        tmp[0 * 4 + i] = s03 + s12
        tmp[1 * 4 + i] = 2 * d03 + d12
        tmp[2 * 4 + i] = s03 - s12
        tmp[3 * 4 + i] = d03 - 2 * d12
    for i in range(4):
        s03 = tmp[i * 4 + 0] + tmp[i * 4 + 3]
        s12 = tmp[i * 4 + 1] + tmp[i * 4 + 2]
        d03 = tmp[i * 4 + 0] - tmp[i * 4 + 3]
        d12 = tmp[i * 4 + 1] - tmp[i * 4 + 2]

        result[i * 4 + 0] = s03 + s12
        result[i * 4 + 1] = 2 * d03 + d12
        result[i * 4 + 2] = s03 - s12
        result[i * 4 + 3] = d03 - 2 * d12

    return result


def IDCT4x4(data):
    tmp = np.zeros((16,),dtype=np.int32)
    d = np.zeros((16,), dtype=np.int32)
    for i in range(4):
        s02 = data[0 * 4 + i] + data[2 * 4 + i]
        d02 = data[0 * 4 + i] - data[2 * 4 + i]
        s13 = data[1 * 4 + i] + (data[3 * 4 + i] >> 1)
        d13 = (data[1 * 4 + i] >> 1) - data[3 * 4 + i]
        tmp[i * 4 + 0] = s02 + s13
        tmp[i * 4 + 1] = d02 + d13
        tmp[i * 4 + 2] = d02 - d13
        tmp[i * 4 + 3] = s02 - s13
    for i in range(4):
        s02 = tmp[0 * 4 + i] + tmp[2 * 4 + i]
        d02 = tmp[0 * 4 + i] - tmp[2 * 4 + i]
        s13 = tmp[1 * 4 + i] + (tmp[3 * 4 + i] >> 1)
        d13 = (tmp[1 * 4 + i] >> 1) - tmp[3 * 4 + i]
        d[0 * 4 + i] = ((s02 + s13 + 32) >> 6)
        d[1 * 4 + i] = ((d02 + d13 + 32) >> 6)
        d[2 * 4 + i] = ((d02 - d13 + 32) >> 6)
        d[3 * 4 + i] = ((s02 - s13 + 32) >> 6)
    d = d.clip(0,255)
    return d

def  DCT8x8(data):
    tmp = np.zeros((64,),dtype=np.int32)
    result = np.zeros((64,),dtype=np.int32)
    for i in range(8):
        #cast byte to int to bepass error "overflow encountered in ubyte_scalars"
        p0 = int(data[i * 8 + 0])
        p1 = int(data[i * 8 + 1])
        p2 = int(data[i * 8 + 2])
        p3 = int(data[i * 8 + 3])
        p4 = int(data[i * 8 + 4])
        p5 = int(data[i * 8 + 5])
        p6 = int(data[i * 8 + 6])
        p7 = int(data[i * 8 + 7])
        a0 = p0 + p7
        a1 = p1 + p6
        a2 = p2 + p5
        a3 = p3 + p4
        b0 = a0 + a3
        b1 = a1 + a2
        b2 = a0 - a3
        b3 = a1 - a2
        a0 = p0 - p7
        a1 = p1 - p6
        a2 = p2 - p5
        a3 = p3 - p4
        b4 = a1 + a2 + ((a0 >> 1) + a0)
        b5 = a0 - a3 - ((a2 >> 1) + a2)
        b6 = a0 + a3 - ((a1 >> 1) + a1)
        b7 = a1 - a2 + ((a3 >> 1) + a3)
        tmp[i * 8 + 0] = b0 + b1
        tmp[i * 8 + 1] = b4 + (b7 >> 2)
        tmp[i * 8 + 2] = b2 + (b3 >> 1)
        tmp[i * 8 + 3] = b5 + (b6 >> 2)
        tmp[i * 8 + 4] = b0 - b1
        tmp[i * 8 + 5] = b6 - (b5 >> 2)
        tmp[i * 8 + 6] = (b2 >> 1) - b3
        tmp[i * 8 + 7] = (b4 >> 2) - b7
    for i in range(8):
        p0 = tmp[0 * 8 + i]
        p1 = tmp[1 * 8 + i]
        p2 = tmp[2 * 8 + i]
        p3 = tmp[3 * 8 + i]
        p4 = tmp[4 * 8 + i]
        p5 = tmp[5 * 8 + i]
        p6 = tmp[6 * 8 + i]
        p7 = tmp[7 * 8 + i]
        a0 = p0 + p7
        a1 = p1 + p6
        a2 = p2 + p5
        a3 = p3 + p4
        b0 = a0 + a3
        b1 = a1 + a2
        b2 = a0 - a3
        b3 = a1 - a2
        a0 = p0 - p7
        a1 = p1 - p6
        a2 = p2 - p5
        a3 = p3 - p4
        b4 = a1 + a2 + ((a0 >> 1) + a0)
        b5 = a0 - a3 - ((a2 >> 1) + a2)
        b6 = a0 + a3 - ((a1 >> 1) + a1)
        b7 = a1 - a2 + ((a3 >> 1) + a3)
        result[0 * 8 + i] = b0 + b1
        result[1 * 8 + i] = b4 + (b7 >> 2)
        result[2 * 8 + i] = b2 + (b3 >> 1)
        result[3 * 8 + i] = b5 + (b6 >> 2)
        result[4 * 8 + i] = b0 - b1
        result[5 * 8 + i] = b6 - (b5 >> 2)
        result[6 * 8 + i] = (b2 >> 1) - b3
        result[7 * 8 + i] = (b4 >> 2) - b7
    return result

def  DCT8x8x264(data):
    tmp = np.zeros((64,),dtype=np.int32)
    result = np.zeros((64,),dtype=np.int32)
    for i in range(8):
        p0 = int(data[i * 8 + 0])
        p1 = int(data[i * 8 + 1])
        p2 = int(data[i * 8 + 2])
        p3 = int(data[i * 8 + 3])
        p4 = int(data[i * 8 + 4])
        p5 = int(data[i * 8 + 5])
        p6 = int(data[i * 8 + 6])
        p7 = int(data[i * 8 + 7])
        s07 = p0 + p7
        s16 = p1 + p6
        s25 = p2 + p5
        s34 = p3 + p4
        a0 = s07 + s34
        a1 = s16 + s25
        a2 = s07 - s34
        a3 = s16 - s25
        d07 = p0 - p7
        d16 = p1 - p6
        d25 = p2 - p5
        d34 = p3 - p4
        a4 = d16 + d25 + (d07 + (d07>>1))
        a5 = d07 - d34 - (d25 + (d25>>1))
        a6 = d07 + d34 - (d16 + (d16>>1))
        a7 = d16 - d25 + (d34 + (d34>>1))
        tmp[i * 8 + 0] = a0 + a1
        tmp[i * 8 + 1] = a4 + (a7>>2)
        tmp[i * 8 + 2] = a2 + (a3>>1)
        tmp[i * 8 + 3] = a5 + (a6>>2)
        tmp[i * 8 + 4] = a0 - a1
        tmp[i * 8 + 5] = a6 - (a5>>2)
        tmp[i * 8 + 6] = (a2>>1) - a3
        tmp[i * 8 + 7] = (a4>>2) - a7
    for i in range(8):
        p0 = tmp[0 * 8 + i]
        p1 = tmp[1 * 8 + i]
        p2 = tmp[2 * 8 + i]
        p3 = tmp[3 * 8 + i]
        p4 = tmp[4 * 8 + i]
        p5 = tmp[5 * 8 + i]
        p6 = tmp[6 * 8 + i]
        p7 = tmp[7 * 8 + i]
        s07 = p0 + p7
        s16 = p1 + p6
        s25 = p2 + p5
        s34 = p3 + p4
        a0 = s07 + s34
        a1 = s16 + s25
        a2 = s07 - s34
        a3 = s16 - s25
        d07 = p0 - p7
        d16 = p1 - p6
        d25 = p2 - p5
        d34 = p3 - p4
        a4 = d16 + d25 + (d07 + (d07 >> 1))
        a5 = d07 - d34 - (d25 + (d25 >> 1))
        a6 = d07 + d34 - (d16 + (d16 >> 1))
        a7 = d16 - d25 + (d34 + (d34 >> 1))
        result[0 * 8 + i] = a0 + a1
        result[1 * 8 + i] = a4 + (a7 >> 2)
        result[2 * 8 + i] = a2 + (a3 >> 1)
        result[3 * 8 + i] = a5 + (a6 >> 2)
        result[4 * 8 + i] = a0 - a1
        result[5 * 8 + i] = a6 - (a5 >> 2)
        result[6 * 8 + i] = (a2 >> 1) - a3
        result[7 * 8 + i] = (a4 >> 2) - a7
    return result

def IDCT8x8(data):
    tmp = np.zeros((64,), dtype=np.int32)
    result = np.zeros((64,), dtype=np.int32)
    for i in range(8):
        p0 = data[i * 8 + 0]
        if i == 0:
            #p0 = p0 + 32
            pass
        p1 = data[i * 8 + 1]
        p2 = data[i * 8 + 2]
        p3 = data[i * 8 + 3]
        p4 = data[i * 8 + 4]
        p5 = data[i * 8 + 5]
        p6 = data[i * 8 + 6]
        p7 = data[i * 8 + 7]
        a0 = p0 + p4
        a1 = p0 - p4
        a2 = p6 - (p2 >> 1)
        a3 = p2 + (p6 >> 1)
        b0 = a0 + a3
        b2 = a1 - a2
        b4 = a1 + a2
        b6 = a0 - a3
        a0 = -p3 + p5 - p7 - (p7 >> 1)
        a1 = p1 + p7 - p3 - (p3 >> 1)
        a2 = -p1 + p7 + p5 + (p5 >> 1)
        a3 = p3 + p5 + p1 + (p1 >> 1)
        b1 = a0 + (a3 >> 2)
        b3 = a1 + (a2 >> 2)
        b5 = a2 - (a1 >> 2)
        b7 = a3 - (a0 >> 2)
        tmp[i * 8 + 0] = b0 + b7
        tmp[i * 8 + 1] = b2 - b5
        tmp[i * 8 + 2] = b4 + b3
        tmp[i * 8 + 3] = b6 + b1
        tmp[i * 8 + 4] = b6 - b1
        tmp[i * 8 + 5] = b4 - b3
        tmp[i * 8 + 6] = b2 + b5
        tmp[i * 8 + 7] = b0 - b7
    for i in range(8):
        p0 = tmp[0 * 8 + i]
        p1 = tmp[1 * 8 + i]
        p2 = tmp[2 * 8 + i]
        p3 = tmp[3 * 8 + i]
        p4 = tmp[4 * 8 + i]
        p5 = tmp[5 * 8 + i]
        p6 = tmp[6 * 8 + i]
        p7 = tmp[7 * 8 + i]
        a0 = p0 + p4
        a1 = p0 - p4
        a2 = p6 - (p2 >> 1)
        a3 = p2 + (p6 >> 1)
        b0 = a0 + a3
        b2 = a1 - a2
        b4 = a1 + a2
        b6 = a0 - a3
        a0 = -p3 + p5 - p7 - (p7 >> 1)
        a1 = p1 + p7 - p3 - (p3 >> 1)
        a2 = -p1 + p7 + p5 + (p5 >> 1)
        a3 = p3 + p5 + p1 + (p1 >> 1)
        b1 = a0 + (a3 >> 2)
        b7 = a3 - (a0 >> 2)
        b3 = a1 + (a2 >> 2)
        b5 = a2 - (a1 >> 2)
        result[0 * 8 + i] = ((b0 + b7) >> 6)
        result[1 * 8 + i] = ((b2 - b5) >> 6)
        result[2 * 8 + i] = ((b4 + b3) >> 6)
        result[3 * 8 + i] = ((b6 + b1) >> 6)
        result[4 * 8 + i] = ((b6 - b1) >> 6)
        result[5 * 8 + i] = ((b4 - b3) >> 6)
        result[6 * 8 + i] = ((b2 + b5) >> 6)
        result[7 * 8 + i] = ((b0 - b7) >> 6)
    result = result.clip(0,255)
    return result

def IDCT8x8x264(data):
    tmp = np.zeros((64,), dtype=np.int32)
    result = np.zeros((64,), dtype=np.int32)
    for i in range(8):
        p0 = int(data[i * 8 + 0])
        if i == 0:
            #p0 = p0 + 32
            pass
        p1 = int(data[i * 8 + 1])
        p2 = int(data[i * 8 + 2])
        p3 = int(data[i * 8 + 3])
        p4 = int(data[i * 8 + 4])
        p5 = int(data[i * 8 + 5])
        p6 = int(data[i * 8 + 6])
        p7 = int(data[i * 8 + 7])
        a0 = p0 + p4
        a2 = p0 - p4
        a4 = (p2>>1) - p6
        a6 = (p6>>1) + p2
        b0 = a0 + a6
        b2 = a2 + a4
        b4 = a2 - a4
        b6 = a0 - a6
        a1 = -p3 + p5 - p7 - (p7>>1)
        a3 =  p1 + p7 - p3 - (p3>>1)
        a5 = -p1 + p7 + p5 + (p5>>1)
        a7 =  p3 + p5 + p1 + (p1>>1)
        b1 = (a7>>2) + a1
        b3 = a3 + (a5>>2)
        b5 = (a3>>2) - a5
        b7 = a7 - (a1>>2)
        tmp[i * 8 + 0] = b0 + b7
        tmp[i * 8 + 1] = b2 + b5
        tmp[i * 8 + 2] = b4 + b3
        tmp[i * 8 + 3] = b6 + b1
        tmp[i * 8 + 4] = b6 - b1
        tmp[i * 8 + 5] = b4 - b3
        tmp[i * 8 + 6] = b2 - b5
        tmp[i * 8 + 7] = b0 - b7
    for i in range(8):
        p0 = tmp[0 * 8 + i]
        p1 = tmp[1 * 8 + i]
        p2 = tmp[2 * 8 + i]
        p3 = tmp[3 * 8 + i]
        p4 = tmp[4 * 8 + i]
        p5 = tmp[5 * 8 + i]
        p6 = tmp[6 * 8 + i]
        p7 = tmp[7 * 8 + i]
        a0 = p0 + p4
        a2 = p0 - p4
        a4 = (p2 >> 1) - p6
        a6 = (p6 >> 1) + p2
        b0 = a0 + a6
        b2 = a2 + a4
        b4 = a2 - a4
        b6 = a0 - a6
        a1 = -p3 + p5 - p7 - (p7 >> 1)
        a3 = p1 + p7 - p3 - (p3 >> 1)
        a5 = -p1 + p7 + p5 + (p5 >> 1)
        a7 = p3 + p5 + p1 + (p1 >> 1)
        b1 = (a7 >> 2) + a1
        b3 = a3 + (a5 >> 2)
        b5 = (a3 >> 2) - a5
        b7 = a7 - (a1 >> 2)
        result[0 * 8 + i] = (b0 + b7)>>6
        result[1 * 8 + i] = (b2 + b5)>>6
        result[2 * 8 + i] = (b4 + b3)>>6
        result[3 * 8 + i] = (b6 + b1)>>6
        result[4 * 8 + i] = (b6 - b1)>>6
        result[5 * 8 + i] = (b4 - b3)>>6
        result[6 * 8 + i] = (b2 - b5)>>6
        result[7 * 8 + i] = (b0 - b7)>>6
    result = result.clip(0,255)
    return result

def blocks_to_yuv(data,w,h):
    assert (len(data) % w*h) == 0
    assert w % 8 == 0
    assert h % 8 == 0
    result = np.zeros((len(data)+ len(data)//2,),dtype=np.uint8)
    z = zr = 0
    print(len(data))
    while (z < len(data)):
        y = 0
        x = 0
        yr = xr = 0
        while (y < h):
            while(x < w):
                i = 0
                while i<8:
                    result[zr + (y+i)*w + x : zr + (y+i)*w + x + 8] = data[z + xr: z + xr + 8]
                    i = i + 1
                    xr = xr + 8
                x = x + 8
            x = 0
            y = y + 8
        zr = zr + w*h + w*h//2
        z = z + w*h
    return result

def yuv_to_blocks(data,w,h):
    assert (len(data) % w * h) == 0
    assert w % 8 == 0
    assert h % 8 == 0
    result = np.zeros((len(data) + len(data) // 2,), dtype=np.uint8)
    z = zr = 0
    print(len(data))
    while (zr < len(data)):
        y = 0
        x = 0
        yr = xr = 0
        while (y < h):
            while (x < w):
                i = 0
                while i < 8:
                    result[z + xr: z + xr + 8] = data[zr + (y + i) * w + x: zr + (y + i) * w + x + 8]
                    i = i + 1
                    xr = xr + 8
                x = x + 8
            x = 0
            y = y + 8
        zr = zr + w * h + w * h // 2
        z = z + w * h
        if z == 51200:
            pass
    return result

'''
    cell_size - maximum count of values of cell
    cell_count - count of cells
    
    :return (average dc, max ac)
'''
def dct_core_init8(cell_size, cell_count):
    t = np.ones((64,),dtype=np.int32)
    t = t * 128
    t = DCT8x8(t)
    t = IDCT8x8(t)
    t = DCT8x8(t)
    middle = t[0]
    a = np.zeros((64,),dtype=np.int32)
    a[0] = middle
    i = 1
    lastmax = 0
    while True:
        for j in range(1,cell_count+1):
            a[j] = i

        b = IDCT8x8(a)
        '''
        a2 = DCT8x8(b)
        b2 = IDCT8x8(a2)
        a3 = DCT8x8(b2)
        b3 = IDCT8x8(a3)
        a4 = DCT8x8(b3)
        b4 = IDCT8x8(a4)
        '''
        max = np.amax(b)
        min = np.amin(b)

        if max == 255:
            break
        if min == 0:
            break
        i = i + 1
    max = i - 1

    i = 1
    j = 0
    while max > 1:
        i = i * 2
        max = max // 2
        j = j + 1
    if  j >= cell_size:
        max = i
        return (middle, max)
    else:
        return None

def cell_to_core(cell,size,core):
    return cell * 2 * core // (size-1)  - core

def cell_to_dct(data,size,core):
    result = np.array(data,dtype=np.int32)
    result = result[:] * 2 * core
    result = result[:] // ( (1 << (size))-1)
    result = result[:] - core
    return result

def dct_to_cell(data,size,core):
    result = np.array(data, dtype=np.float32)
    result = result[:] + core
    result = result[:] *  ( (1 << (size))-1)
    result = result[:] / (2 * core)
    result = np.clip(result,0,(1<<size)-1)
    result = np.round(result)
    result = result.astype(np.int32)
    return result

def dct_to_block(data, core_middle, count):
    assert ((data.shape[0] % count) == 0)
    result = np.zeros((data.shape[0]*64//count), dtype=np.uint8)
    dct = np.zeros(64, dtype=np.int32)
    assert (len(data) % count) == 0
    for i in range(len(data)//count):
        dct[0] = core_middle
        result[64 * i] = core_middle
        for j in range(count):
            #ii = i*count + j
            #iii = 64*i+j+1
            #result[64*i+j+1] = data[i*count + j]
            dct[j+1] = data[i*count + j]
        dct2 = dct[_snake_encode8_plain_np]
        block = IDCT8x8x264(dct2)
        result[64*i:64*i+64] = block[:]
    return result

def block_to_dct(data,count):
    assert ((data.shape[0] % 64)==0)
    result = np.zeros((data.shape[0] * count // 64), dtype=np.int32)
    idct = np.empty(count,dtype=np.int32)
    for i in range(len(data)//64):
        a = data[64*i:64*i+64]
        b = DCT8x8x264(a)
        c = b[_snake_decode8_plain_np]
        #dct = DCT8x8(data[64*i:64*i+64])[_snake_decode8_plain_np]
        d = c
        dct = d[1: 1+count]
        #result[i*count: (i+1)*count] = dct[1: 1+count]
        # value shift compensation
        dct = dct# + dct // 7
        result[i * count: (i + 1) * count] = dct

    return result



'''
generate possible modes

:return list of dict 
  count - count of dct coefficients 
  size - "alphabet" size
  volume - total data per dct block
'''
def make_dct_match_list():
    modes = {[]:'origin'}
    cnt = 0
    print("Generating possible modes:")
    np_dct = np.empty((64,),dtype=np.int16)
    for count in range(1, 64):
        for size in [2, 4, 8, 16]:
            t = dct_core_init8(size, count)
            if t is None:
                continue
            middle, core = t
            np_dct.fill(0)
            np_dct[0] = middle
            flag = True
            for variant in itertools.permutations(range(size), count):
                _ = list(map(lambda x: cell_to_core(x, size, core), variant))
                __ = cell_to_dct(variant, size, core)
                np_dct[1:1 + count] = __[:]
                np_idct = IDCT8x8(np_dct)
                np_dct2 = DCT8x8(np_idct)
                variant_2 = dct_to_cell(np_dct2[1:1 + count], size, core)
                if not (variant == variant_2).all():
                    flag = False
                    break
            if flag:
                print(f" count={count}  size={size}  volume={count*size}")
                modes.append({'count': count, 'size': size, 'volume':count*size})





class Feeder:
    def __init__(self, chunk_size, dtype=np.uint8):
       self._chunk_size = chunk_size
       self._size = 0
       self._list = []
       self._is_open = True
       self._dtype = dtype

    def iter(self):
        while self._is_open:
            if self._size >= self._chunk_size:
                size = ( self._size // self._chunk_size) * self._chunk_size
                i = 0
                j = 0
                buf = np.zeros((size,),dtype=self._dtype)
                while (i < size) and (j < len(self._list)):
                    ssize = len(self._list[j])
                    if i + ssize < size:
                      buf[i:i+ssize] = self._list[j][:]
                      j = j + 1
                      i = i + ssize
                    else:
                      wsize = ssize - ((i + ssize) - size)
                      buf[i:i+wsize] = self._list[j][:wsize]
                      self._list[j] = self._list[j][wsize:]
                      i = i + wsize
                if j > 0:
                    self._list = self._list[j:]
                self._size = self._size - size
                yield buf
            else:
                pass

        #after Feeder closed - flush all data with padding zeros
        if self._size > 0:
            size = (self._size // self._chunk_size) * self._chunk_size
            if (self._size % self._chunk_size) > 0:
                size = size + self._chunk_size
            buf = np.zeros((size,), dtype=self._dtype)
            i = 0
            for data in self._list:
                buf[i:i+len(data)] = data[:]
                test = data[:]
                buf[i:i + len(data)] = test[:]
            self._list = []
            self._size = 0
            yield buf

    def push(self,data):
        self._list.append(data)
        self._size = self._size + len(data)

    def close(self):
        self._is_open = False

    def have_item(self):
        if self._is_open:
            return (self._size >= self._chunk_size)
        else:
            return self._size > 0


'''
y = np.random.randn(16, 16)*50
y = y.astype(dtype=np.uint8)
ydct = dctn(y, norm='ortho')
yidct = idctn(ydct,norm='ortho')

print(np.allclose( y, idctn(dctn(y, norm='ortho'), norm='ortho')) )
'''
def array_print(data,x):
    f = open(f"test {x}.txt","w")
    for i in range(len(data)):
        f.write(f"{i}  {data[i]}\n")
    f.close()

def array_save(data,x):
    f = open(f"save 0{x*10}.txt","bw")
    f.write(memoryview(data))
    f.close()

def array_saved(data,x):
    f = open(f"saved 0{x*10}.txt","bw")
    f.write(memoryview(data))
    f.close()

def array_debug_compare(x,y,s):
    f = open(f"debug 0{s}.txt", "w")
    m = x.shape[0] if x.shape[0] < y.shape[0] else y.shape[0]
    for i in range(m):
        f.write(f"i{i}  a{int(x[i])-int(y[i])}  x{x[i]}  y{y[i]}\n")
    f.close()

def array_read():
    return np.fromfile(open("decode.yuv","rb"),dtype=np.uint8)


cells_per_block = 12
cells_size = 1
w = 8*20
h = 8*20
yuv_size = w*h + w*h//2
middle, core_val = dct_core_init8(cells_size,cells_per_block)


np.random.seed(2)
data = np.random.randint(0,255,600,dtype=np.uint8)
data_rs = bytes_to_rs(data)
data_unrs = bytes_from_rs(data_rs)
array_save(data,1)
data01 = data
#data = bytes_to_rs(data)
data_size = len(data)

data = bytes_to_bits(data)
array_save(data,2)
data02 = data

buf = Feeder(cells_size)
buf.push(data)
buf.close()
data = bits_to_cell(next(buf.iter()),cells_size)
array_save(data,3)
data03 = data

data = cell_to_dct(data,cells_size,core_val)
array_save(data,4)
data04 = data

#array_print(data,'7dct')

buf = Feeder(cells_per_block,dtype=np.int32)
buf.push(data)
buf.close()
data = dct_to_block(next(buf.iter()), middle, cells_per_block)
array_save(data,5)
data05 = data

#array_print(data,'8block')

buf = Feeder( w*h )
buf.push(data)
buf.close()
data = next(buf.iter())
data = blocks_to_yuv(data,w,h)
array_save(data,6)
data06 = data

#data = np.fromfile(open("save 060.txt","rb"),dtype=np.uint8)


#array_print(data,'9end')


data = array_read()
array_saved(data,6)
data16 = data
array_debug_compare(data06,data16,6)

data = yuv_to_blocks(data,w,h)
array_saved(data,5)
data15 = data
array_debug_compare(data05,data15,5)

data = block_to_dct(data,cells_per_block)
array_saved(data,4)
data14 = data
array_debug_compare(data04,data14,4)

data = dct_to_cell(data,cells_size,core_val)
array_saved(data,3)
data13 = data
array_debug_compare(data03,data13,3)

data = cell_to_bits(data,cells_size)
array_saved(data,2)
data12 = data
array_debug_compare(data02,data12,2)


buf = Feeder(8,dtype=np.uint8)
buf.push(data)
buf.close()
data = bits_to_bytes(next(buf.iter()))
array_saved(data,1)
#data = bytes_from_rs(data[:data_size])

data11 = data
array_debug_compare(data01,data11,1)

'''class Buffer():

    def __init__(self,size,dtype):
        self._buff = np.zeros((size,),dtype=dtype)
        self._size = 0
        self._limit = size
        self._ia = 0
        self._ib = 0
        self._is_closed = False

    @property
    def size(self):
        return self._size

    @property
    def is_closed(self):
        return self._is_closed


    def put(self,ndarray):
        if self.is_closed:
            raise AttributeError
        size = ndarray.shape[0]
        while(self._limit < ( self._size + size )):
            sleep(0.1)

        if (self._ib + size) < self._limit:
            pass
    def get(self,size):
        if self.is_closed and self.size==0:
            raise AttributeError
        while((self.size<size) and not self.is_closed):
            sleep(0.1)
        if self.is_closed:
            if self.size < size:
                return self.read(size) + bytearray(size - self.size)
        else:
            return self.read(size)
'''

