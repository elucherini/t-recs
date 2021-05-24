"""
Classes for logging
"""
import logging
from abc import ABC


class VerboseMode(ABC):
    """Abstract class for verbose mode"""

    def __init__(self, name, verbose=False):
        self._logger = DebugLogger(name, verbose)

    def __del__(self):
        """
        Closes the logging handler upon garbage collection.
        """
        self.close()

    def set_verbose(self, toggle):
        """Toggle verbosity"""
        try:
            self._logger.set_verbose(toggle)
        except TypeError as err:
            print("set_verbose:", err)

    def is_verbose(self):
        """Return True if verbosity is enabled, False otherwise"""
        return self._logger.is_verbose()

    def log(self, msg):
        """Log given message"""
        self._logger.log(msg)

    def close(self):
        """Close the logging file handler"""
        # occasionally verbose objects are created incidentally
        # when performing matrix operations
        if hasattr(self, "_logger"):
            self._logger.handler.close()
            self._logger.logger.removeHandler(self._logger.handler)


class DebugLogger:
    """Class to configure debug logging module"""

    def __init__(self, name, verbose=False):
        """Instantiate DebugLogger object
        @name: name of logger
        @level: level of logger (see documentation of logging module)
        """
        # create logger
        self.logger = logging.getLogger(name)
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        self.logger.setLevel(level)

        # create file handler and set level to debug
        self.handler = logging.FileHandler("trecs.log")
        self.handler.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(name)s - %(message)s")

        # add formatter to handler
        self.handler.setFormatter(formatter)

        # add handler to logger
        self.logger.addHandler(self.handler)

        # test
        self._test_configured_logger()

    def _test_configured_logger(self):
        """Simple test to announce logger is enabled"""
        self.logger.debug("Debugging module initialized")

    def log(self, message):
        """Log at DEBUG level"""
        self.logger.debug(message)

    def is_verbose(self):
        """Return True if debugger is enabled; That is, if debugger can log
        DEBUG-level messages
        """
        return self.logger.isEnabledFor(logging.DEBUG)

    def set_verbose(self, verbose=False):
        """Enable/disable verbose"""
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool, got %s type" % type(verbose))
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO

        self.logger.setLevel(level)
