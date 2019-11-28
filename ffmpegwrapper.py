from subprocess import Popen, PIPE, DEVNULL
import numpy as np
import sys


class ffmpeg:
    def __init__(self, cmdln, use_stdin=False, use_stdout=False, use_stderr=False, print_to_console=False):
        self._process = None
        # self._use_stdin = use_stdin
        # self._use_stdout = use_stdout
        # self._use_stderr = use_stderr
        self._cmdln = cmdln
        self._stdin = None
        if use_stdin:
            self._stdin = PIPE
        self._stdout = None
        self._stderr = None
        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        self._process = Popen(
            self._cmdln
            , stdin=self._stdin
            , stdout=self._stdout
            , stderr=self._stderr
        )

    # read  cnt bytes as np array uint8
    def readout(self, cnt):
        buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)
        #print(arr.shape)
        return arr

    def readerr(self, cnt):
        buf = self._process._stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None
