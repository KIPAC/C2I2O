c2i2o: Cosmology to Intermediates to Observables
=================================================

**c2i2o** is a Python library for cosmological parameter inference and emulation.

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.12+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

Overview
--------

c2i2o provides a unified framework for bidirectional transformations in
cosmological analysis:

.. math::

   \text{Cosmological Parameters} \longleftrightarrow \text{Intermediate Data Products} \longleftrightarrow \text{Observables}

Key Features
------------

* **Fast Emulation**: Replace expensive simulations with trained emulators
* **Flexible Inference**: Multiple inference backends (MCMC, nested sampling, SBI)
* **Extensible**: Plugin architecture for custom emulators and observables
* **Multi-Framework**: Interfaces to CCL, Astropy, PyTorch, TensorFlow
* **Scalable**: Designed for diverse cosmological datasets

Quick Start
-----------

Install c2i2o:

.. code-block:: bash

   pip install c2i2o

Basic usage:

.. code-block:: python

   from c2i2o.core import CosmologicalParameters

   # Define cosmological parameters
   params = CosmologicalParameters(
       omega_m=0.3,
       omega_b=0.05,
       h=0.7,
       sigma_8=0.8,
       n_s=0.96
   )

   print(params)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
