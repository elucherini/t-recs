import numpy as np
from abc import ABC, abstractmethod
from rec.utils import VerboseMode
import inspect

class FromNdArray(np.ndarray, VerboseMode):
    """Subclass for Numpy's ndarrays
    """
    def __new__(cls, input_array, num_items=None, verbose=False):
        obj = np.asarray(input_array).view(cls)
        obj.verbose = verbose
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.verbose = getattr(obj, 'verbose', False)


class BaseObserver(ABC):
    """Observer mixin for the observer design pattern
    """
    def register_observables(self, **kwargs):
        observables = kwargs.pop("observables", None)

        observer = kwargs.pop("observer", None)
        if observer is None:
            raise ValueError("Argument `observer` cannot be None")
        elif not isinstance(observer, list):
            raise TypeError("Argument `observer` must be a list")

        observable_type = kwargs.pop("observable_type", None)
        if not inspect.isclass(observable_type):
            raise TypeError("Argument `observable_type` must be a class")

        self._add_observable(observer=observer, observables=observables,
                            observable_type=observable_type)

    def _add_observable(self, observer, observables, observable_type):
        if len(observables) < 1:
            raise ValueError("Can't add fewer than one observable!")
        new_observables = list()
        for observable in observables:
            if isinstance(observable, observable_type):
                new_observables.append(observable)
            else:
                raise ValueError("Observables must be of type %s" % observable_type)
        observer.extend(new_observables)


class BaseObservable(ABC):
    """Observable mixin for the observer design patter
    """
    def get_observable(self, **kwargs):
        data = kwargs.pop("data", None)
        if data is None:
            raise ValueError("Argument `data` cannot be None")
        elif not isinstance(data, list):
            raise TypeError("Argument `data` must be a list")
        if len(data) > 0:
            name = getattr(self, 'name', 'Unnamed')
            return {name: data}
        else:
            return None

    def observe(self, *args, **kwargs):
        pass


class BaseComponent(BaseObservable, VerboseMode, ABC):
    def __init__(self, verbose=False, init_value=None):
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.state_history = list()
        self.state_history.append(init_value)

    def get_component_state(self):
        return self.get_observable(data=self.state_history)

    @abstractmethod
    def store_state(self):
        pass

    def get_timesteps(self):
        return len(self.state_history)


class Component(FromNdArray, BaseComponent):
    def __init__(self, current_state=None, size=None, verbose=False, seed=None):
        # general input checks
        if current_state is not None:
            if not isinstance(current_state, (list, np.ndarray)):
                raise TypeError("current_state must be a list or numpy.ndarray")
        if current_state is None and size is None:
            raise ValueError("current_state and size can't both be None")
        if current_state is None and not isinstance(size, tuple):
            raise TypeError("size must be a tuple, is %s" % type(size))
        if current_state is None and size is not None:
            current_state = Generator(seed).binomial(n=.3, p=1, size=size)
        self.current_state = current_state
        # Initialize component state
        BaseComponent.__init__(self, verbose=verbose, init_value=self.current_state)

    def store_state(self):
        self.state_history.append(np.copy(self.current_state))
