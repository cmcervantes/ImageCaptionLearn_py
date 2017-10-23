import logging
import sys
from time import time
from os.path import abspath, expanduser

__author__ = "ccervantes"


class Logger:
    """
    Logging functionality wrapper
    """

    def __init__(self, lvl, delay, filename=None):
        """
        Initializes the logger with the default logging
        level and status delay; optional argument filename
        specifies whether to log to a file or standard out
        :param lvl: Default logging level
        :param delay: Time in seconds between writing
                      status messages
        :param filename: File to which output should be logded
        """
        self._prev_ts = 0.0
        self._tic = 0.0
        self._delay = delay

        self._levels = dict()
        levels = Logger.get_logging_lvls()
        for i in range(len(levels)):
            self._levels[levels[i]] = 10 * (i+1)
        #endfor

        # Set up the logger depending on whether
        # we're writing to the console
        self._logger = logging.getLogger()
        if filename is None:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(filename)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s (%(levelname)s): %(message)s', '%H:%M:%S'))
        self._logger.addHandler(handler)
        self._logger.setLevel(self._levels[lvl])
    #enddef

    @staticmethod
    def get_logging_lvls():
        """
        Returns the logging levels
        """
        return ['debug', 'info', 'warning', 'error', 'critical']
    #enddef

    def debug(self, msg=None, format_str=None, *format_args):
        """
        Logs a debug message; message can be explicit or in the
        printf format with associated args
        """
        if msg is not None:
            self.__log('debug', msg)
        elif format_str is not None:
            self.__log('debug', format_str % format_args)
    #enddef

    def info(self, msg=None, format_str=None, *format_args):
        """
        Logs an info message; message can be explicit or in the
        printf format with associated args
        """
        if msg is not None:
            self.__log('info', msg)
        elif format_str is not None:
            self.__log('info', format_str % format_args)
    #enddef

    def warning(self, msg=None, format_str=None, *format_args):
        """
        Logs a warning message; message can be explicit or in the
        printf format with associated args
        """
        if msg is not None:
            self.__log('warning', msg)
        elif format_str is not None:
            self.__log('warning', format_str % format_args)
    #enddef

    def error(self, msg=None, format_str=None, *format_args):
        """
        Logs an error message; message can be explicit or in the
        printf format with associated args
        """
        if msg is not None:
            self.__log('error', msg)
        elif format_str is not None:
            self.__log('error', format_str % format_args)
    #enddef

    def critical(self, msg=None, format_str=None, *format_args):
        """
        Logs a critical message; message can be explicit or in the
        printf format with associated args
        """
        if msg is not None:
            self.__log('critical', msg)
        elif format_str is not None:
            self.__log('critical', format_str % format_args)
    #enddef

    def __log(self, lvl, msg):
        """
        Sends the specified message to the logger at the given level
        """
        self._logger.log(level=self._levels[lvl], msg=msg)
    #enddef

    def log_status(self, lvl, msg=None, format_str=None, *format_args):
        """
        Logs a status message to the console, using the specified
        message or printf structure. Status messages differ from
        normal logs in that they are only printed if
        self._delay seconds have passed since the last status message
        """
        current_ts = time()

        if self._prev_ts < 1 :
            self._prev_ts = current_ts
        if current_ts >= self._prev_ts + self._delay:
            if format_str is not None:
                msg = format_str % format_args
            self.__log(lvl, msg)
            self._prev_ts = current_ts
        #endif
    #enddef

    def tic(self, lvl=None, msg=None):
        """
        Logs 'tic'; sets the internal tic
        value to now; meant to be used with toc()
        """
        self._tic = time()
        if lvl is not None and msg is not None:
            msg += " (tic)"
        else:
            lvl = 'debug'
            msg = "tic"
        #endif
        self.__log(lvl, msg)
    #enddef

    def toc(self, lvl=None):
        """
        Logs "toc" with the time in seconds since
        tic was called
        """
        if lvl is None:
            lvl = 'debug'
        self.__log(lvl, "toc (%.3fs)" % (time() - self._tic))
    #enddef
#endclass