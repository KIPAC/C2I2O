## Charge

Help me to set up and implement python classes a machine learning project.


## Project parameters


### Input data

The input data set will be a small number (5-10) of parameters.   The represent physical quantities and have physical bounds.


### Output data (also called "intermediates")

For each different set of parameter values we can generate a number of output data arrays.  We are going to call these output data arrays "intermediates".

There will be a number of different intermediates.   In general these will be 1 or 2 dimensional arrays of fixed (but configurable) size.   The one dimensional arrays
will be evaluated on a grid of values we refer to as "scale factors" or "a".   The two dimensional arrays will be evaluated on different grid of scale factors on one axis, and a grid
of wavenumbers or "k" on the other axis.   The units for wave number are inverse Mpc.   We expect the length both the scale factor and wavenumber grid to be a few hundred.

For any given set of input parameters we can use external code to exactly compute any of the intermediates.

### Emulation

We would like to train up an emulators that can accurately reproduce the intermediates.


### Interfence

We would like to be able use a set of intermediates to infer the input paramters.  We would like this inference to be flexible to different possible sets of intermediates, and tolerant of missing data.


### Software tools

We would like to do this with three different softare tools:

1. pytorch
2. tensorflow
3. a symbolic regression algorithm


### Software design parameters

We would like the follow design parameters

1. use python >= 3.12
2. use type hints
3. use pytdantic and yaml for configuration
4. reduce code duplication and prioritize maintainability


### Project paramters

1. Set up base classes for emulation and inference that will work with all there sets of software tools.
2. Implement all there different version


## Communication Guidelines

- Ask clarifying questions when requirements are ambiguous
- Provide context for design decisions
- Suggest best practices and alternatives
- Flag potential issues early
- Keep documentation synchronized with code changes
