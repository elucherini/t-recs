T-RECS (Tool for RecSys Simulation)
=====================================

.. image:: https://i.imgur.com/3ZRDVaD.png
  :width: 400
  :alt: Picture of T-Rex next to letters T-RECS

A library for using agent-based modeling to perform simulation studies of sociotechnical systems.

Installation
------------

System requirements
###################

Currently, the simulator has only been tested extensively on MacOS 10.15 and Ubuntu 20.04.
This simulator supports Python 3.7+ and it has not been tested with older versions of Python 3. If you have not configured a Python environment locally, please follow Scipy's `instructions for installing a scientific Python distribution`_.

.. _instructions for installing a scientific Python distribution: https://scipy.org/install.html

If you do not have Python 3.7+ installed, you can create a new environment with Python 3.7 by running the following command in terminal:

..  code-block:: bash

    conda create --name py37 python=3.7

To ensure the example Jupyter notebooks run in your Python 3.7 environment, follow `the instructions from this blog post`_. **Note**: you will also need ``pandas`` to run the example notebooks. As of December 2020, we recommend installing ``pandas v1.0.5`` using the command: ``pip install 'pandas==1.0.5'``. This will help avoid package conflicts between ``pandas`` and ``pylint`` if you also plan on contributing to ``trecs`` and running tests.

.. _the instructions from this blog post: https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

For users
#########

To install the simulator, you will need the Python package manager, ``pip``. After activating your virtual environment, run the following command in a terminal:

..  code-block:: bash

  pip install trecs

For developers
##############

If you'd like to install the latest version of ``trecs`` based on what is currently in the main branch of the repository, run the following commands after activating your virtual environment:

..  code-block:: bash

  git clone https://github.com/elucherini/t-recs.git
  cd t-recs
  pip install -e .

Additionally, you may run ``pip install -r requirements-dev.txt`` to install a few additional dependencies that will be useful for linting, testing, etc.

Documentation
**************
If you would like to edit the documentation, see the ``docs/`` folder. To build the documentation on your local folder, you will need to install ``sphinx`` and the ``sphinx-rtd-theme`` via ``pip``. Next, ``cd`` into the ``docs`` folder and run ``make html``. The output of the command should tell you where the compiled HTML documentation is located.

.. _sphinx: https://www.sphinx-doc.org/en/master/usage/installation.html
.. _sphinx-rtd-theme: https://pypi.org/project/sphinx-rtd-theme/

Tutorials
----------
Examples of how to use the simulator can be found in the notebooks below:

- `Quick start`_: start here for a brief introduction.
- `Complete guide`_: an overview of the main concepts of the system.
- Advanced guide - `building a model`_: an introduction to adding your own models on top of the system.
- Advanced guide - `adding metrics`_: an example of how to add new metrics to a model.

.. _Quick start: https://github.com/elucherini/t-recs/blob/main/examples/quick-start.ipynb
.. _Complete guide: https://github.com/elucherini/t-recs/blob/main/examples/complete-guide.ipynb
.. _building a model: https://github.com/elucherini/t-recs/blob/main/examples/advanced-models.ipynb
.. _adding metrics: https://github.com/elucherini/t-recs/blob/main/examples/advanced-metrics.ipynb

Please check the examples_ directory for more notebooks.

.. _examples: examples/

Example usage
-------------

..  code-block:: bash

  import trecs

  recsys = trecs.models.ContentFiltering()
  recsys.run(timesteps=10)
  measurements = recsys.get_measurements()

Documentation
--------------

A first draft of the documentation is available `here`_. In its current version, the documentation can be used as a supplement to exploring details in the code. Currently, the tutorials in examples_ might be a more useful and centralized resource to learn how to use the system.

.. _here: https://elucherini.github.io/t-recs/index.html
.. _examples: examples/


Contributing
--------------

Thanks for your interest in contributing! Check out the guidelines for contributors in `CONTRIBUTING.md`_.

.. _CONTRIBUTING.md: https://github.com/elucherini/t-recs/blob/main/CONTRIBUTING.md
