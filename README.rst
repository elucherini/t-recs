Simulator
=========
A simulator for sociotechnical systems. More details coming soon.

Install
-------

This simulator supports Python 3.7+ and it has not been tested with older versions of Python 3. If you have not configured a Python environment locally, please follow Scipy's `instructions for installing a scientific Python distribution`_.

.. _instructions for installing a scientific Python distribution: https://scipy.org/install.html

Currently, the simulator has only been tested extensively on MacOS 10.15.

To install the simulator, you will need the Python package manager, pip. After activating your virtual environment, run the following commands in a terminal:

..  code-block:: bash

  git clone https://github.com/elucherini/algo-segregation.git
  cd algo-segregation
  pip install -e .


The command should automatically install all dependencies.

Tutorials
----------
Examples of how to use the simulator can be found in the notebooks below:

- `Quick start`_: start here for a brief introduction.
- `Complete guide`_: an overview of the main concepts of the system.
- Advanced guide - `building a model`_: an introduction to adding your own models on top of the system.
- Advanced guide - `adding metrics`_: an example of how to add new metrics to a model.

.. _Quick start: examples/quick-start.ipynb
.. _Complete guide: examples/complete-guide.ipynb
.. _building a model: examples/advanced-models.ipynb
.. _adding metrics: examples/advanced-metrics.ipynb

Please check the examples_ directory for more notebooks.

.. _examples: examples/

Example usage
-------------

..  code-block:: bash

  import rec

  recsys = rec.models.ContentFiltering()
  recsys.run(timesteps=10)
  measurements = recsys.get_measurements()
  
Documentation
--------------

A first draft of the documentation is available `here`_. In its current version, the documentation can be used as a supplement to exploring details in the code. Currently, the tutorials in examples_ might be a more useful and centralized resource to learn how to use the system.

.. _here: https://elucherini.github.io/algo-segregation/index.html
.. _examples: examples/


