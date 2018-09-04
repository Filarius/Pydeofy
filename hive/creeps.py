import multiprocessing as mp
from .message import Message,MessageType
from multiprocessing import Pipe,Process,Manager
import time

class Larve:
    # (function, Pipe with Message, Queue size, Queue size)
    def __init__(self,body,in_size=5,out_size=5):
        self._body = body
        self._input = mp.Queue(in_size)
        self._output = mp.Queue(out_size)

        def f(self):
            #is_last = False
            while True:
                if self._input.empty():
                    time.sleep(0.1)
                    continue

                m = self._input.get()
                #m = Message()

                if m.type == MessageType.EOF:
                    break

                if m.type == MessageType.Data:
                    data = self._body(m.body)
                    mes = Message(m.id, MessageType.Data, data)
                    self._output.put(mes)


        self._process = Process(target=f,args=(self,))

    def start(self):
        self._process.start()

    def is_alive(self):
        self._process.is_alive()

    def is_hungry(self):
        return not self._input.full()

    def is_pooped(self):
        return not self._output.empty()

    def put(self,message):
        self._input.put(message)

    def get(self):
        return self._output.get()

    def eof(self):
        self._input.put(Message(0,MessageType.EOF,None))

    def kill(self): #TO DO - how to kill worker
        self._input.close()
        while not self._input.empty():
            self._input.get()

        self._output.close()
        while not self._output.empty():
            self._output.get()

        self._process.terminate()



class LarvePool():
    def __init__(self,body,size,lifespan=None,in_mem=10,out_mem=10,sort=True):
        self._body = body
        self._size = size
        self._lifespan = lifespan
        self._sort = sort
        self._input = mp.Queue(in_mem)
        self._output = mp.Queue(out_mem)
        buf = mp.Queue()

        def callback(message):
            buf.put(message)

        def func(self, message):
            data = self._body(message.body)
            return Message(message.id, message.type, data)

        def loop(self):
            eof = False
            idfirst = None
            pool = mp.Pool(processes=self._size, maxtasksperchild=self._lifespan)
            msgbuf = []
            while True:
                m = self._input.get()
                m = Message()

                if m.type == MessageType.EOF:
                    eof = True

                elif m.type == MessageType.Data:
                    if self._sort:
                        if idfirst==None:
                            idfirst = m.id

                    pool.apply_async(self._body,(self,m),callback=callback)

                if self._sort:
                    updated = False
                    while not buf.empty():
                        updated = True
                        msgbuf.append(buf.get())
                    if updated:
                        msgbuf.sort(key= lambda m: -m.id)
                        while (len(msgbuf)>0) and (msgbuf[-1].id == idfirst):
                            self._output.put(msgbuf.pop(-1))
                else:
                    while not buf.empty():
                        self._output.put(buf.get())

        def ishungry():
            return not self._input.full()

        def ispooped():
            return  not self._output.empty()

        def get():
            return self._output.get()

        def put(message):
            self._input.put(message)




