import logging
import sys
from time import time
from os.path import abspath, expanduser

"""
Wrapper for logging functionality

@author ccervantes
"""
class LogUtil:
    def __init__(self, lvl, delay, filename=None):
        self._prevTS = 0.0
        self._tic = 0.0
        self._delay = delay

        self._levels = dict()
        levels = LogUtil.get_logging_lvls()
        for i in range(len(levels)):
            self._levels[levels[i]] = 10 * (i+1)
        #endfor

        outStream = sys.stdout
        if filename is not None:
            outStream = open(abspath(expanduser(filename)), 'w')

        logging.basicConfig(level=self._levels[lvl],
            format='%(asctime)s (%(levelname)s): %(message)s',
            datefmt='%H:%M:%S', stream=outStream)
    #enddef

    """
    Returns the logging levels
    """
    @staticmethod
    def get_logging_lvls():
        return ['debug', 'info', 'warning', 'error', 'critical']
    #enddef

    """
    Logs a debug message; message can be explicit or in the
    printf format with associated args
    """
    def debug(self, msg=None, format_str=None, *format_args):
        if msg is not None:
            self.__log('debug', msg)
        elif format_str is not None:
            self.__log('debug', format_str % format_args)
    #enddef

    """
    Logs an info message; message can be explicit or in the
    printf format with associated args
    """
    def info(self, msg=None, format_str=None, *format_args):
        if msg is not None:
            self.__log('info', msg)
        elif format_str is not None:
            self.__log('info', format_str % format_args)
    #enddef

    """
    Logs a warning message; message can be explicit or in the
    printf format with associated args
    """
    def warning(self, msg=None, format_str=None, *format_args):
        if msg is not None:
            self.__log('warning', msg)
        elif format_str is not None:
            self.__log('warning', format_str % format_args)
    #enddef

    """
    Logs an error message; message can be explicit or in the
    printf format with associated args
    """
    def error(self, msg=None, format_str=None, *format_args):
        if msg is not None:
            self.__log('error', msg)
        elif format_str is not None:
            self.__log('error', format_str % format_args)
    #enddef

    """
    Logs a critical message; message can be explicit or in the
    printf format with associated args
    """
    def critical(self, msg=None, format_str=None, *format_args):
        if msg is not None:
            self.__log('critical', msg)
        elif format_str is not None:
            self.__log('critical', format_str % format_args)
    #enddef

    """
    Sends the specified message to the logger at the given level
    """
    def __log(self, lvl, msg):
        logging.log(level=self._levels[lvl], msg=msg)
    #enddef

    """
    Logs a status message to the console, using the specified
    message or printf structure. Status messages differ from
    normal logs in that they are only printed if
    self._delay seconds have passed since the last status message
    """
    def log_status(self, lvl, msg=None, format_str=None, *format_args):
        currentTS = time()

        if self._prevTS < 1 :
            self._prevTS = currentTS
        if currentTS >= self._prevTS + self._delay:
            if format_str is not None:
                msg = format_str % format_args
            self.__log(lvl, msg)
            self._prevTS = currentTS
        #endif
    #enddef

    """
    Logs 'tic'; sets the internal tic
    value to now; meant to be used with toc()
    """
    def tic(self, lvl=None, msg=None):
        self._tic = time()
        if lvl is not None and msg is not None:
            msg += " (tic)"
        else:
            lvl = 'debug'
            msg = "tic"
        #endif
        self.__log(lvl, msg)
    #enddef

    """
    Logs "toc" with the time in seconds since
    tic was called
    """
    def toc(self, lvl=None):
        if lvl is None:
            lvl = 'debug'
        self.__log(lvl, "toc (%.3fs)" % (time() - self._tic))
    #enddef
#endclass