import math,numpy as np
#бинарное кодирование
class BinaryCoder:
    # (максимальное число)
    def __init__(self,val_count):
        self._count = val_count
        self._width = int(math.ceil(math.log2(val_count)-0.00000001))
        self._dtype = None

        choiseX = [np.uint8,np.uint16,np.uint32,np.uint64]
        choiseY = [8,16,32,64]
        for x,y in zip(choiseX,choiseY):
            if self._width <= y:
                self._dtype = x
                self._width = y
                break

    #  количество бит на число
    def get_width(self):
        return  self._width
    def get_dtype(self):
        return self._dtype

    # (массив numpy int)
    def encode(self,arr):
        arr = arr.astype(self._dtype)
        shape = arr.shape[0]
        if self._width > 8:
            arr = arr.view('uint8')
        a = np.unpackbits(arr)
        a = a.reshape(shape,self._width)
        return a

    # (массив numpy float)
    def decode(self,arr):
        #пастеризация
        a = np.array(arr>0.5,dtype=np.uint8)
        # декодинг
        a = np.packbits(a)
        if self._width > 8:
            a = a.view('uint{0}'.format(self._width))
        return a