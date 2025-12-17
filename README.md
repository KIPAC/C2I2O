# c2i2o: Cosmology to Intermediates to Observables

[![Tests](https://github.com/KIPAC/c2i2o/workflows/tests/badge.svg)](https://github.com/KIPAC/c2i2o/actions)
[![Documentation](https://github.com/KIPAC/c2i2o/workflows/Documentation/badge.svg)](https://github.com/KIPAC/c2i2o/actions)
[![PyPI version](https://badge.fury.io/py/c2i2o.svg)](https://badge.fury.io/py/c2i2o)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/c2i2o.svg)](https://pypi.python.org/pypi/c2i2o/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A modern Python library for cosmological parameter inference and emulation.

## Overview

**c2i2o** provides a unified framework for bidirectional transformations in cosmological analysis:

Cosmological Parameters to/from Intermediate Data Products to/from Observables

### Key Features

- **🚀 Fast Emulation**: Replace expensive simulations with trained emulators
- **📊 Flexible Inference**: Multiple inference backends (MCMC, nested sampling, SBI)
- **🔌 Extensible**: Plugin architecture for custom emulators and observables
- **🌐 Multi-Framework**: Interfaces to CCL, Astropy, PyTorch, TensorFlow
- **📈 Scalable**: Designed for diverse cosmological datasets

## Installation

### Basic Installation

```bash
pip install c2i2o
```

# For PyTorch-based emulators
```bash
pip install c2i2o[pytorch]
```

# For TensorFlow-based emulators
```bash
pip install c2i2o[tensorflow]
```

# For cosmology library integrations
```bash
pip install c2i2o[cosmology]
```

# For database support
```bash
pip install c2i2o[database]
```

# Install everything
```bash
pip install c2i2o[all]
```



git clone https://github.com/KIPAC/c2i2o.git
cd c2i2o
pip install -e ".[dev]"


## Documentation

Full documentation is available at [c2i2o.readthedocs.io](https://c2i2o.readthedocs.io).

Build documentation locally:

```bash
cd docs
make html
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
```
