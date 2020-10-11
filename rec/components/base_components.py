import numpy as np
from abc import ABC, abstractmethod
from rec.utils import VerboseMode
import inspect


class FromNdArray(np.ndarray, VerboseMode):
    """Subclass for Numpy's ndarrays."""

    def __new__(cls, input_array, num_items=None, verbose=False):
        obj = np.asarray(input_array).view(cls)
        obj.verbose = verbose
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.verbose = getattr(obj, "verbose", False)


class BaseObserver(ABC):
    """Observer mixin for the observer design pattern."""

    def register_observables(self, observables=None, observer=None, observable_type=None):
        if observer is None:
            raise ValueError("Argument `observer` cannot be None")
        elif not isinstance(observer, list):
            raise TypeError("Argument `observer` must be a list")

        if not inspect.isclass(observable_type):
            raise TypeError("Argument `observable_type` must be a class")

        self._add_observables(
            observer=observer, observables=observables, observable_type=observable_type
        )

    def unregister_observables(self, observables=None, observer=None):
        if observer is None:
            raise ValueError("Argument `observer` cannot be None")
        elif not isinstance(observer, list):
            raise TypeError("Argument `observer` must be a list")

        self._remove_observables(observer=observer, observables=observables)

    def _add_observables(self, observer, observables, observable_type):
        if len(observables) < 1:
            raise ValueError("Can't add fewer than one observable!")
        new_observables = list()
        for observable in observables:
            if isinstance(observable, observable_type):
                new_observables.append(observable)
            else:
                raise ValueError("Observables must be of type %s" % observable_type)
        observer.extend(new_observables)

    def _remove_observables(self, observer, observables_to_remove):
        if len(observables) < 1:
            raise ValueError("Can't remove fewer than one observable!")
        observables_copy = observer.copy()
        for observable in observables_to_remove:
            if observable in observables_copy:
                observables_copy.remove(observable)
            else:
                raise ValueError("Cannot find %s!" % observable)
        observer = observables_copy


class BaseObservable(ABC):
    """Observable mixin for the observer design pattern."""

    def get_observable(self, **kwargs):
        data = kwargs.pop("data", None)
        if data is None:
            raise ValueError("Argument `data` cannot be None")
        elif not isinstance(data, list):
            raise TypeError("Argument `data` must be a list")
        if len(data) > 0:
            name = getattr(self, "name", "Unnamed")
            return {name: data}
        else:
            return None

    @abstractmethod
    def observe(self, *args, **kwargs):
        pass


class BaseComponent(BaseObservable, VerboseMode, ABC):
    """Observable that stores a history of its state."""

    def __init__(self, verbose=False, init_value=None):
        VerboseMode.__init__(self, __name__.upper(), verbose)
        self.state_history = list()
        if isinstance(init_value, np.ndarray):
            init_value = np.copy(init_value)
        self.state_history.append(init_value)

    def get_component_state(self):
        return self.get_observable(data=self.state_history)

    def observe(self, state, copy=True):
        if copy:
            to_append = np.copy(state)
        else:
            to_append = state
        self.state_history.append(to_append)

    def get_timesteps(self):
        return len(self.state_history)


class Component(FromNdArray, BaseComponent):
    """Class for components that make up the system state."""

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
            current_state = Generator(seed).binomial(n=0.3, p=1, size=size)
        self.current_state = current_state
        # Initialize component state
        BaseComponent.__init__(self, verbose=verbose, init_value=self.current_state)

    def store_state(self):
        self.observe(self, copy=True)


if __name__ == "__main__":
    from rec.models.recommender import SystemStateModule
    from rec.components import PredictedUserProfiles
    import numpy as np

    class Test(SystemStateModule):
        def __init__(self, user_profiles):
            self.user_profiles = PredictedUserProfiles(user_profiles)
            SystemStateModule.__init__(self)
            self.add_state_variable(self.user_profiles)

        def run(self, to_add=1):
            self.user_profiles += to_add
            print("Adding %d to user profiles. Result:\n%s\n\n" % (to_add, self.user_profiles))
            self.measure_content()

        def measure_content(self):
            for component in self._system_state:
                component.store_state(np.asarray(component))

    profiles = np.zeros((5, 5))
    test = Test(profiles)
    test.run()
    test.run()
    print("State history:")
    print(test.user_profiles.state_history)
    print("Final user profiles")
    print(test.user_profiles)
    print(test.user_profiles.name)
