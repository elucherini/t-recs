import logging
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Class containing dictionary of DebugLoggers
class Debug():
    '''
    '' Create and set up dictionary of loggers
    '' @names: list of strings, or string
    '' @enabled: list of bools, or bool
    '' Assume names and enabled has the same
    ''     number of elements
    '''
    def __init__(self, names, enabled=False):
        self.logger = dict()
        # If only one logger
        if not isinstance(names, list):
            # This determines whether to enable debugging log
            if enabled is True:
                level = logging.DEBUG
            else:
                level = logging.INFO
            self.logger[names] = DebugLogger(names, level)
            return
        # If multiple loggers (common case)
        for name, en in zip(names, enabled):
            if en is True:
                level = logging.DEBUG
            else:
                level = logging.INFO
            self.logger[name] = DebugLogger(name, level)

    '''
    '' Return the desired DebugLogger object
    '' @name: key of desired logger in dictionary
    '''
    def get_logger(self, name):
        return self.logger[name]

    def flush_all(self):
        for _, name in enumerate(self.logger.keys()):
            self.logger[name].handler.flush()

# Class to configure debug logging module
class DebugLogger():
    '''
    '' @name: name of logger
    '' @level: level of logger (see documentation of logging module)
    '''
    def __init__(self, name, level=logging.INFO):
        # create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # create console handler and set level to debug
        self.handler = logging.StreamHandler()
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
    def is_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)

    '''
    '' Return True if logger can print the final results
    '' of the simulation.
    '''
    def can_show_results(self):
        return self.logger.isEnabledFor(logging.INFO)

    '''
    '' Wrapper around PyPlot's plot functions
    '' it supports PyPlot's plotting functions and the most used properties
    '''
    def pyplot_plot(self, *plot_args, plot_func=plt.plot,
                    title=None, xlabel=None, ylabel=None):
        plt.style.use('seaborn-whitegrid')
        plot_func(*plot_args)
        if title is not None:
            plt.title(title)
            log_msg = title
        else:
            log_msg = ''
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        self.log('Plot: ' + log_msg)
        plt.show()

    # TODO implement
    '''
    '' Wrapper around Seaborn's plot functions
    '''
    #def seaborn_plot(self, plot_func=sns., *plot_args, title=None):
    #   return