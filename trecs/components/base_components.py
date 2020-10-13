""" Base components that are the building blocks of measuring and tracking
    variables of interest in the recommender systems
 """
import inspect
from abc import ABC, abstractmethod
import numpy as np
from trecs.logging import VerboseMode
from trecs.random import Generator


class FromNdArray(np.ndarray, VerboseMode):
    """Subclass for Numpy's ndarrays."""

    def __new__(cls, input_array, verbose=False):
        obj = np.asarray(input_array).view(cls)
        obj.verbose = verbose
        return obj

    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        pass

    def __array_finalize__(self, obj):
        """ Set the verbosity based on the object passed in """
        if obj is None:
            return
        self.verbose = getattr(  # pylint: disable=attribute-defined-outside-init
            obj, "verbose", False
        )


# Observer methods for the observer design pattern
def register_observables(observer, observables=None, observable_type=None):
    """Add items in observables to observer list"""
    if not inspect.isclass(observable_type):
        raise TypeError("Argument `observable_type` must be a class")

    if len(observables) < 1:
        raise ValueError("Can't add fewer than one observable!")

    new_observables = list()
    for observable in observables:
        if isinstance(observable, observable_type):
            new_observables.append(observable)
        else:
            raise ValueError(f"Observables must be of type {observable_type}")
    observer.extend(new_observables)


def unregister_observables(observer, observables):
    """Remove items in observables from observer list"""
    if len(observables) < 1:
        raise ValueError("Can't remove fewer than one observable!")
    for observable in observables:
        if observable in observables:
            observer.remove(observable)
        else:
            raise ValueError("Cannot find %s!" % observable)


class BaseObservable(ABC):
    """Observable mixin for the observer design pattern."""

    def get_observable(self, **kwargs):
        """ Returns the value of this observable as a dict """
        data = kwargs.pop("data", None)
        if data is None:
            raise ValueError("Argument `data` cannot be None")
        if not isinstance(data, list):
            raise TypeError("Argument `data` must be a list")
        if len(data) > 0:
            name = getattr(self, "name", "Unnamed")
            return {name: data}
        return None

    @abstractmethod
    def observe(self, *args, **kwargs):
        """ Abstract method that should involve "recording" the observable """


class BaseComponent(BaseObservable, VerboseMode, ABC):
    """Observable that stores a history of its state."""

    def __init__(self, verbose=False, init_value=None):
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.state_history = list()
        if isinstance(init_value, np.ndarray):
            init_value = np.copy(init_value)
        self.state_history.append(init_value)

    def get_component_state(self):
        """ Return the history of the component's values as a dictionary """
        return self.get_observable(data=self.state_history)

    def observe(self, state, copy=True):  # pylint: disable=arguments-differ
        """ Append the current value of the variable (by default a copy) to the
            state history """
        if copy:
            to_append = np.copy(state)
        else:
            to_append = state
        self.state_history.append(to_append)

    def get_timesteps(self):
        """ Get the number of timesteps in the state history """
        return len(self.state_history)


class Component(FromNdArray, BaseComponent):
    """Class for components that make up the system state."""

    def __init__(
        self, current_state=None, size=None, verbose=False, seed=None
    ):  # pylint: disable=super-init-not-called
        # general input checks
        if current_state is not None:
            if not isinstance(current_state, (list, np.ndarray)):
                raise TypeError("current_state must be a list or numpy.ndarray")
        if current_state is None and size is None:
            raise ValueError("current_state and size can't both be None")
        if current_state is None and not isinstance(size, tuple):
            raise TypeError("size must be a tuple, is %s" % type(size))
        if current_state is None and size is not None:
            current_state = Generator(seed).binomial(n=0.3, p=1, size=size)
        self.current_state = current_state
        # Initialize component state
        BaseComponent.__init__(self, verbose=verbose, init_value=self.current_state)

    def store_state(self):
        """ Store a copy of the component's value in the state history """
        self.observe(self, copy=True)


class SystemStateModule:  # pylint: disable=too-few-public-methods
    """
    Mixin for observers of :class:`Component` observables. Implements the
    `Observer design pattern`_.

    .. _`Observer design pattern`: https://en.wikipedia.org/wiki/Observer_pattern

    This mixin allows the system to monitor the system state. That is, at each
    timestep, an element will be added to the
    :attr:`~components.base_components.BaseComponent.state_history` lists
    of each component that the system is monitoring.

    Attributes
    ------------

        _system_state: list
            List of system state components that the system will monitor.

    """

    def __init__(self):
        self._system_state = list()

    def add_state_variable(self, *args):
        """
        Adds metrics to the :attr:`_system_state` list. This allows the system
        to monitor these system state components.

        Parameters
        -----------

            args: :class:`~components.base_components.BaseComponent`
                Accepts a variable number of components that inherit from class
                :class:`~components.base_components.BaseComponent`
        """
        register_observables(
            observer=self._system_state, observables=list(args), observable_type=BaseComponent,
        )
