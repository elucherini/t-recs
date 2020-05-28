# Simulator
A simulator for sociotechnical systems. More details coming soon.

## Install

This simulator supports Python 3.7+ and it has not been tested with older versions of Python 3. If you have not configured a Python environment locally, please follow Scipy's [instructions for installing a scientific Python distribution](https://scipy.org/install.html).

Currently, the simulator has only been tested extensively on MacOS 10.15.

To install the simulator, you will need the Python package manager, pip. After activating your virtual environment, run the following commands in a terminal:

```
      git clone https://github.com/elucherini/algo-segregation.git
      cd algo-segregation
      python3 -m pip install -e rec
```

The command should automatically install all dependencies.

## Tutorials
Examples of how to use the simulator can be found in the notebooks below:

- [Quick start](examples/quick-start.ipynb): start here for a brief introduction.
- [Complete guide](examples/complete-guide.ipynb): an overview of the main concepts of the system.
- [Advanced guide: models](examples/advanced-models.ipynb): an introduction to adding your own models on top of the system.
- [Advanced guide: metrics](examples/advanced-metrics.ipynb): an example of how to add new metrics to a model.

## Example usage

```
import rec

recsys = rec.models.ContentFiltering()
recsys.run(timesteps=10)
measurements = recsys.get_measurements()
```
