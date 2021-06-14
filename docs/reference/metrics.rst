Metrics
=============

**Metrics** are components that track and store a given quantity over the duration of the simulation. All metrics in this module are based on the :class:`~metrics.measurement.Measurement` abstract class, which inherits from the :class:`~components.base_components.BaseObservable` abstract class (refer to the `Observer design pattern`_ for more details).

This module comes with a number of pre-loaded metrics, as well as a simple programmer interface to develop new metrics.

.. _Observer design pattern: https://en.wikipedia.org/wiki/Observer_pattern

Interaction Measurement
------------------------

.. autoclass:: metrics.measurement.InteractionMeasurement
  :members:

Interaction Spread
------------------------

.. autoclass:: metrics.measurement.InteractionSpread
  :members:

RecSimilarity
------------------------

.. autoclass:: metrics.measurement.RecSimilarity
  :members:

InteractionSimilarity
------------------------

.. autoclass:: metrics.measurement.InteractionSimilarity
  :members:

MSE Measurement
------------------------

.. autoclass:: metrics.measurement.MSEMeasurement
  :members:

Diffusion Tree
------------------------

.. autoclass:: metrics.measurement.DiffusionTreeMeasurement
  :members:

Measurement of Structural Virality
-----------------------------------

.. autoclass:: metrics.measurement.StructuralVirality
  :members:

Average Feature Score Range
-----------------------------------

.. autoclass:: metrics.measurement.AverageFeatureScoreRange
  :members:

Measurement: base class
------------------------

.. autoclass:: metrics.measurement.Measurement
  :members:
