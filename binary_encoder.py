import math,numpy as np
#бинарное кодирование
class BinaryCoder:
    # (максимальное число)
    def __init__(self,width):
        self._width = width

    #  количество бит на число
    def get_width(self):
        return  self._width

    # (массив numpy int)
    def encode(self,arr):

        a = np.unpackbits(arr)
        a = a.reshape(-1,self._width)
        return a

    # (массив numpy float)
    def decode(self,arr):
        #пастеризация
        a = np.array(arr>0.5,dtype=np.uint8)
        # декодинг
        a = np.packbits(a)
        #if self._width > 8:
        #    a = a.view('uint{0}'.format(self._width))
        return a