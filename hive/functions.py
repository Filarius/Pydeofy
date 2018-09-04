import numpy as np
import multiprocessing
import io
from time import sleep
from hive import fifo





b = io.StringIO()

b.write("33")
print("2",b.read(2))

a = fifo.Fifo()
a.write(b"1234")
a.write(b"5678")
a.write(b"9abcd")
print(a.read(88))









def bytes_to_bits(arr):
    arr = np.array(arr)
    return np.unpackbits(arr).tobytes()

def  bits_to_bytes(arr):
    if (len(arr) % 8) != 0:
        raise AttributeError
    arr = np.array(arr)
    return np.packbits(arr).tobytes()

def  bits_to_cell(arr,size):
    if (len(arr) % size) != 0:
        raise AttributeError
    for i in range(len(arr)):
        




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

_snake_encode8_plain = [x + y*8 for x,y in _snake_encode8]

_snake_encode4 = [(0,0),
(0,1),(1,0),
(2,0),(1,1),(0,2),
(0,3),(1,2),(2,1),(3,0),
(3,1),(2,2),(1,3),
(2,3),(3,2),
(3,3)
]

_snake_encode4_plain = [x + y*4 for x,y in _snake_encode4]

#def YFrameTo



def snake_encode8(arr,chunksize):
#place values from chunks into 8x8 flat arrays with snake order
    assert (arr.shape[0] % chunksize) == 0

    shape = (arr.shape[0] / chunksize, 64)
    temp = np.zeros(arr.shape,dtype=np.uint8)
    i=0

    while i<= arr.shape[0]:
        j = 0
        while j < chunksize:
            temp[i + _snake_encode8_plain[j]]
            j += 1
        i += chunksize
    chunk_count = arr.shape[0] / chunksize
    temp.reshape(chunk_count * 64)
    return temp

def snake_decode8(arr,chunksize):
    assert  (arr.shape[0] % 64) == 0

    chunk_count = arr.shape[0] /64



class Buffer():

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