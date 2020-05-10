import logging
import sys
import numpy as np
from abc import ABC, abstractmethod

# Abstract class for verbose mode
class VerboseMode(ABC):

    def __init__(self, name, verbose=False):
        self._logger = DebugLogger(name, verbose)

    '''
    '' Toggle verbosity
    '' @toggle: True/False
    '''
    def set_verbose(self, toggle):
        try:
            self._logger.set_verbose(toggle)
        except TypeError as e:
            print("set_verbose:", e)

    '''
    '' Return True if verbosity is enabled,
    '' False otherwise
    '''
    def is_verbose(self):
        return self._logger.is_verbose()

    def log(self, msg):
        self._logger.log(msg)


# Class to configure debug logging module
class DebugLogger():
    '''
    '' @name: name of logger
    '' @level: level of logger (see documentation of logging module)
    '''
    def __init__(self, name, verbose=False):
        # create logger
        self.logger = logging.getLogger(name)
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        self.logger.setLevel(level)

        # create file handler and set level to debug
        self.handler = logging.FileHandler('rec.log')
        self.handler.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(name)s - %(message)s')

        # add formatter to handler
        self.handler.setFormatter(formatter)

        # add handler to logger
        self.logger.addHandler(self.handler)

        # test
        self._test_configured_logger()

    '''
    '' Simple test to announce logger is enabled
    '''
    def _test_configured_logger(self):
        self.logger.debug("Debugging module inizialized")

    '''
    '' Log at DEBUG level
    '' @message: message to log
    '''
    def log(self, message):
        self.logger.debug(message)

    '''
    '' Return True if debugger is enabled
    '' That is, if debugger can log DEBUG-level messages
    '''
    def is_verbose(self):
        return self.logger.isEnabledFor(logging.DEBUG)

    '''
    '' Enable/disable verbose
    '' @verbose: bool
    '''
    def set_verbose(self, verbose=False):
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool, got %s type" % type(verbose))
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO

        self.logger.setLevel(level)


# Test
if __name__ == '__main__':
    class A(VerboseMode):
        def __init__(self):
            self.a = 2
            super().__init__(__name__, True)
            self.log("hi")

    a = A()
    a.log("verbose")
    try:
        a.set_verbose('a')
    except TypeError as e:
        print(e)

    a.log("verbose")
