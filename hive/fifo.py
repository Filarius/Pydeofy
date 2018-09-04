#!/usr/bin/python
"""
A reasonably efficient FIFO buffer.
Copyright (C) 2013  Byron Platt

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


_DISCARD_SIZE = 0xffff


class Fifo(object):
    """
    A reasonably efficient FIFO buffer. Based loosely on the python
    implementation of StringIO.  The general idea is that all writes are
    appended to the end of the buffer and all reads originate at the start of
    the buffer (ie a FIFO.)
    """

    def __init__(self, data=b'', eol='\r\n',size_limit=0xffff):
        """
        Creates a FIFO.

        Args:
            data: Optionally initialize with the contents of a string.
            eol: The EOL marker that will be used when readline() or peekline()
                is used (including using the iterator).
        """
        self.buf = data
        self.eol = eol
        self.buflist = []
        self.pos = 0
        self._discard_size = size_limit

    def __len__(self):
        self.__append()
        return len(self.buf) - self.pos

    def __iter__(self):
        return self

    def __discard(self):
        if self.pos > self._discard_size:
            self.buf = self.buf[self.pos:]
            self.pos = 0

    def __append(self):
        if len(self.buflist)>0:
            self.buf += b''.join(self.buflist)
            self.buflist = []

    def clear(self):
        """
        Clears the FIFO.
        """
        self.buf = ''
        self.buflist = []
        self.pos = 0

    def write(self, data):
        """
        Writes data to the FIFO.
        """
        self.buflist.append(data)

    def read(self, length=-1):
        """
        Reads from the FIFO.

        Reads as much data as possible from the FIFO up to the specified
        length. If the length argument is negative or ommited all data
        currently available in the FIFO will be read. If there is no data
        available in the FIFO an empty string is returned.

        Args:
            length: The amount of data to read from the FIFO. Defaults to -1.
        """
        if 0 <= length < len(self):
            newpos = self.pos + length
            data = self.buf[self.pos:newpos]
            self.pos = newpos
            self.__discard()
            return data

        data = self.buf[self.pos:]
        self.clear()
        return data

    def readblock(self, length=-1):
        """
        Reads a block from the FIFO.

        Reads exactly as much data as given by the length argument from the
        FIFO. If the length argument is negative or ommited all data currently
        available in the FIFO will be read. If there is less data available in
        the FIFO than the amount given by length an empty string is returned.

        Args:
            length: The size of the block to be read from the FIFO. Defaults
                to -1.
        """
        if 0 <= length <= len(self):
            return self.read(length)

        return ''

    def readline(self):
        """
        Reads a line from the FIFO.

        Reads one line from the FIFO. If there are no EOLs in the FIFO then an
        empty string is returned.
        """
        self.__append()

        i = self.buf.find(self.eol, self.pos)
        if i < 0:
            return ''

        newpos = i + len(self.eol)
        data = self.buf[self.pos:newpos]
        self.pos = newpos
        self.__discard()
        return data

    def readuntil(self, token, size=0):
        """
        Reads data from the FIFO until a token is encountered.

        If no token is encountered as much data is read from the FIFO as
        possible keeping in mind that the FIFO must retain enough data to
        perform matches for the token across writes.

        Args:
            token: The token to read until.
            size: The minimum amount of data that should be left in the FIFO.
                This is only used if it is greater than the length of the
                token.  When ommited this value will default to the length of
                the token.

        Returns: A tuple of (found, data) where found is a boolean indicating
            whether the token was found, and data is all the data that could be
            read from the FIFO.

        Note: When a token is found the token is also read from the buffer and
            returned in the data.
        """
        self.__append()

        i = self.buf.find(token, self.pos)
        if i < 0:
            index = max(len(token) - 1, size)
            newpos = max(len(self.buf) - index, self.pos)
            data = self.buf[self.pos:newpos]
            self.pos = newpos
            self.__discard()
            return False, data

        newpos = i + len(token)
        data = self.buf[self.pos:newpos]
        self.pos = newpos
        self.__discard()
        return True, data

    def peek(self, length=-1):
        """
        Peeks into the FIFO.

        Performs the same function as read() without removing data from the
        FIFO. See read() for further information.
        """
        if 0 <= length < len(self):
            newpos = self.pos + length
            return self.buf[self.pos:newpos]

        return self.buf[self.pos:]

    def peekblock(self, length=-1):
        """
        Peeks a block into the FIFO.

        Performs the same function as readblock() without removing data from
        the FIFO. See readblock() for further information.
        """
        if 0 <= length <= len(self):
            return self.peek(length)

        return ''

    def peekline(self):
        """
        Peeks a line into the FIFO.

        Perfroms the same function as readline() without removing data from the
        FIFO. See readline() for further information.
        """
        self.__append()

        i = self.buf.find(self.eol, self.pos)
        if i < 0:
            return ''

        newpos = i + len(self.eol)
        return self.buf[self.pos:newpos]

    def peekuntil(self, token, size=0):
        """
        Peeks for token into the FIFO.

        Performs the same function as readuntil() without removing data from the
        FIFO. See readuntil() for further information.
        """
        self.__append()

        i = self.buf.find(token, self.pos)
        if i < 0:
            index = max(len(token) - 1, size)
            newpos = max(len(self.buf) - index, self.pos)
            return False, self.buf[self.pos:newpos]

        newpos = i + len(token)
        return True, self.buf[self.pos:newpos]

    def next(self):
        """
        Iterates over the FIFO one line at a time.
        """
        line = self.readline()
        if not line:
            raise StopIteration()

        return line
