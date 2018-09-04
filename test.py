# -*- coding: utf-8 -*-
import random
import time
from threading import Thread
from multiprocessing import Process


class MyThread(Thread):
    """
    A threading example
    """

    def __init__(self, name):
        """Инициализация потока"""
        Thread.__init__(self)
        self.name = name

    def run(self):
        """Запуск потока"""
        s = 0
        for i in range(100000000):
            s += i*i

        msg = "%s is running" % s
        print(msg)

def f():
    s=0
    for i in range(100000000):
        s += i * i

    msg = "%s PROCESS" % s
    print(msg)


def create_threads():
    """
    Создаем группу потоков
    """
    for i in range(20):
        name = "Thread #%s" % (i + 1)
        my_thread = MyThread(name)
        my_thread.start()
        #p = Process(target=f)
        #p.start()


if __name__ == "__main__":
    create_threads()