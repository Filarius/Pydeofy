import math
import numpy as np
import sys
import sparse


# обеспечивает конвертацию чисел в массив из нескольких нулей
# и определенным количеством единиц

class MultiHotEncoding:
    def __init__(self, values_cnt, ones_count, dim_extend=0):
        '''
        values_count - количество различных значений
        ones_count - количество единичек
        dim_extend - увеличивает размер последовательности
                    относительно минимально допустимого
        '''

        self._ones_count = ones_count
        self._values_count = values_cnt
        self._row_size = self._get_row_size(values_cnt, ones_count) + dim_extend  # вычисляем количество ячеек в массиве
        #self.row_size = self._row_size
        self._dict = np.zeros((values_cnt, self._row_size),
                              dtype=np.uint8)  # массив-словарь для перевода чисел в нули и единицы
        self._dict_shape = None
        # self._dict_ones_pos_to_index = np.zeros(shape,dtype=np.uint32)
        # print(self._dict_shape)
        #print('row_size ', self._row_size)
        self._dict_idx = {}  # переменная для словаря обратной конвертации
        # список из границы (минимум, максимум) возможных координат соответствующих единиц
        self._idx_bounds = [[self._row_size, 0] for i in range(self._ones_count)]
        self._dict_fill()  # заполняет словари и список границ
        self._dict_to_np()  # конвертирует словарь для кодировки в более подходящий вид

    def _get_row_size(self, vals, ones):
        if ones == 1:
            return vals
        target = vals * math.factorial(ones)
        xmin = 1
        xmax = ones + 1
        p = 1
        for i in range(2, ones + 1):
            p *= i
        while p <= target:
            p = p // xmin
            p *= xmax
            xmin += 1
            xmax += 1
        return xmax - 1

    def _dict_fill(self):
        row = np.array(range(self._ones_count), np.uint64)
        arr = self._dict
        # idx = self._dict_ones_pos_to_index
        cnt = 0
        i = self._ones_count - 1
        bounds = self._idx_bounds
        while cnt < self._values_count:
            arr[cnt, row] = 1
            # idx[row] = cnt
            self._dict_idx[tuple(row)] = cnt
            for i in range(self._ones_count):
                if bounds[i][0] > row[i]:
                    bounds[i][0] = row[i]
                if bounds[i][1] < row[i]:
                    bounds[i][1] = row[i]

            # print(cnt,arr[cnt,:])
            while i >= 0:
                overflow = False
                if i == self._ones_count - 1:
                    if row[i] == self._row_size - 1:
                        if self._ones_count==1:
                            break
                        overflow = True
                else:
                    if row[i] == (row[i + 1] - 1):
                        overflow = True
                if overflow:
                    if i == 0:
                        raise Exception('<=> Error :3 <=>')
                    i -= 1
                else:
                    row[i] += 1
                    while i < self._ones_count - 1:
                        i += 1
                        row[i] = row[i - 1] + 1
                    break
            cnt += 1

    def _dict_to_np(self):
        dct = self._dict_idx
        bounds = self._idx_bounds
        choose = [[2 ** 8, np.uint8],
                  [2 ** 16, np.uint16],
                  [2 ** 32, np.uint32],
                  [2 ** 64, np.uint64],
                  ]
        a_type = None
        for chosen in choose:
            if self._values_count <= chosen[0]:
                a_type = chosen[1]
                break
        if a_type is None:
            raise Exception('<=> ERROR :3 <=>')
        shape = [int(x[1] - x[0] + 1) for x in bounds]
        #print('shape', shape)
        a = np.zeros(shape, dtype=a_type)
        # print('idx bounds ', self._idx_bounds)
        keys = [key for key in self._dict_idx.keys()]
        keys = list(map(list, zip(*keys)))  # list of lists transpose magicks
        values = [key for key in self._dict_idx.values()]
        s = sparse.COO(keys, values, shape=shape).astype(a_type)

        for key, val in self._dict_idx.items():
            i = tuple([[key[i] - self._idx_bounds[i][0]] for i in range(self._ones_count)])
            a[i] = val
        #print(a)
        #a = s.todense()

        # test !
        # dct - dictinary
        # s  - sparse
        # a - numpy
        self._dict_idx = a

    # преобразует массив из чисел в массив из строк с нулями и единицами
    def encode(self, arr):
        return self._dict[arr]

    # преобразует массив из строк с вероятностями в массив чисел которые им соответствуют
    def decode_legacy(self, arr):
        dif = [x[0] for x in self._idx_bounds]# координаты единичек в словаре сдвинуты на значения из dif
        dif = np.array(dif, dtype=np.int32)
        # тут черная магия, в maxargs массив из строк где в строках числа
        # которые показывают координаты единичек, строки отсортированы по возрастанию
        lim = np.full(shape=self._ones_count, fill_value=-self._ones_count, dtype=np.int32)
        maxargs = np.argpartition(arr, lim, axis=1)[:, -self._ones_count:][:, ::-1]
        maxargs = np.sort(maxargs, axis=1)
        maxargs = maxargs - dif
        idxs = self._dict_idx
        ret = [idxs[tuple(i)] for i in maxargs]
        return ret

    def decode(self,arr):
        dif = [x[0] for x in self._idx_bounds]# координаты единичек в словаре сдвинуты на значения из dif
        maxs = np.array([x[1] for x in self._idx_bounds])
        dif = np.array(dif, dtype=np.int32)
        # тут черная магия, в maxargs массив из строк где в строках числа
        # которые показывают координаты единичек, строки отсортированы по возрастанию
        lim = np.full(shape=self._ones_count, fill_value=-self._ones_count, dtype=np.int32)
        maxargs = np.argpartition(arr, lim, axis=1)[:, -self._ones_count:][:, ::-1]
        maxargs = np.sort(maxargs, axis=1)
        maxargs = maxargs - dif
        outbounds = np.max(maxargs,0)
        outbounds = outbounds - maxs
        if np.max(outbounds) >= 0: # есть координаты выходящие за границу массива
            outbounds
            a=  np.argmax(outbounds)

            #b=np.where((maxargs[:,a]>maxs[a]))
            #b = np.where((maxargs > maxs))
            #maxargs[b[0]]=0
            maxargs = np.where((maxargs > maxs),0,maxargs) # ставит нули на неправильных индексах единичек
            a=0


        outbounds = np.max(maxargs, 0)
        outbounds = outbounds - maxs

        idxc = self._dict_idx

        idxi = np.ravel_multi_index(maxargs.transpose(), idxc.shape)

        ret = np.take(idxc, idxi)
        print('000')
        return ret

    def get_width(self):
        return self._row_size




'''
t = MultiHotEncoding(2 ** (8), 3)
size = t.get_width()
print('===========')
a = np.random.randint(256,size=30)
print(a)
b = t.encode(a)
#print(b)
c = t.decode(b)
print(c)
print(c-a)
'''
from timeit import timeit
setup = '''
import numpy as np
from __main__ import MultiHotEncoding
t = MultiHotEncoding(2 ** (16), 8)
print(t.get_width())
size = t.get_width()
#print('===========')
a = np.random.randint(2**16,size=30000)
#print(a)
b = t.encode(a)
#print(b)
#c = t.decode(b)
#print(c)
'''
stmt = '''
c = t.decode(b)
'''
n= 100
#print(timeit(stmt,setup,number=n)/n)



#print('============================')
'''
a = np.arange(27).reshape(3,3,3)*10
ad = {i:v for i,v in np.ndenumerate(a)}
#a = a.reshape(27)
b = np.array([[1,2,1],[2,2,2]])
'''
#C
#print( np.take(a, np.ravel_multi_index(b.transpose(), a.shape)))
#np.take(a, np.ravel_multi_index(indexes.transpose(), a.shape))
#print(np.ravel_multi_index(b, (3,3)))
#[a[tuple(i)] for i in b]

from timeit import timeit


setup = '''
import numpy as np
import struct
pack = struct.Struct('<H').pack
a = np.arange(27).reshape(3,3,3)*5
b = np.array([[1,2,1],[2,2,2]])
ad = {i:v for i,v in np.ndenumerate(a)}

'''
stmt = '''
[ad[tuple(i)] for i in b]
'''
#print(timeit(stmt=stmt,setup=setup))

#print(timeit(stmt=stmt,setup=setup))
stmt = '''
#bytes([a[tuple(i)] for i in b])
#'''
#print(timeit(stmt=stmt,setup=setup))
#stmt = 'np.take(a, np.ravel_multi_index(b.transpose(), a.shape))'
