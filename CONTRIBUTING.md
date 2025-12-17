# Contributing to c2i2o

Thank you for your interest in contributing to c2i2o! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/c2i2o.git
   cd c2i2o
   ```

3. Set up the development environment:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### Making Changes

1. Write your code following the style guidelines
2. Add or update tests as needed
3. Update documentation if necessary
4. Ensure all tests pass:
```bash
pytest
```


## Code Style

### We use several tools to maintain code quality:

1. Black for code formatting (110 char line length)
2. Ruff for linting
3. mypy for type checking

Run these before committing:
```bash
black src/ tests/
ruff check src/ tests/ --fix
mypy src/
```

Or use pre-commit hooks:
```bash
pre-commit run --all-files
```

### Commit Messages

We follow the Conventional Commits format:

```php
<type>(<scope>): <subject>
<body>
<footer>
```

Types:

    feat: New feature
    fix: Bug fix
    docs: Documentation changes
    style: Code style changes (formatting, etc.)
    refactor: Code refactoring
    test: Adding or updating tests
    chore: Maintenance tasks


Examples:
```bash
feat(emulators): add Gaussian process emulator

Implements a GP-based emulator for power spectrum prediction.
Uses scikit-learn's GaussianProcessRegressor.

Closes #42
```

###Testing

    Write tests for all new features and bug fixes
    Aim for ~100% code coverage
    Use pytest fixtures for common test setups
    Test edge cases and error conditions

Run tests:

```bash
# All tests
pytest

# With coverage
pytest --cov=c2i2o

# With coverage report
pytest --cov=c2i2o --cov-report=html

# Specific test file
pytest tests/core/test_parameters.py

# Specific test
pytest tests/core/test_parameters.py::test_cosmological_parameters_validation
```

## Documentation

    1. Update docstrings for all new/modified functions and classes
    2. Use NumPy style docstrings
    3. Include type hints in function signatures
    4. Add examples to docstrings where helpful


Example docstring:

```python
def calculate_power_spectrum(
    k: np.ndarray,
    params: CosmologicalParameters,
    redshift: float = 0.0
) -> np.ndarray:
    """
    Calculate the matter power spectrum at given wavenumbers.

    Parameters
    ----------
    k : np.ndarray
        Wavenumbers in h/Mpc, shape (n_k,)
    params : CosmologicalParameters
        Cosmological parameters
    redshift : float, optional
        Redshift at which to evaluate, by default 0.0

    Returns
    -------
    np.ndarray
        Power spectrum values in (Mpc/h)^3, shape (n_k,)

    Examples
    --------
    >>> params = CosmologicalParameters(omega_m=0.3, h=0.7)
    >>> k = np.logspace(-3, 1, 100)
    >>> pk = calculate_power_spectrum(k, params)
    """
```

## Submitting Changes

    1. Push your branch to your fork:

    ```bash
    git push origin feature/your-feature-name
    ```

    2. Create a Pull Request on GitHub
        Provide a clear title and description
        Reference any related issues
        Ensure all CI checks pass
        Request review from maintainers

    3. Address review feedback
        Be responsive to reviewer comments
        Make requested changes in new commits
        Push updates to your branch
        Re-request review after changes


## Review Process

    1. All submissions require review from a maintainer
    2. CI checks must pass (tests, linting, type checking)
    3. Code coverage should not decrease significantly
    4. Documentation must be updated as needed
    5. Changes should maintain backward compatibility when possible


## Types of Contributions

### Bug Reports

When reporting bugs, please include:

    Clear description of the issue
    Steps to reproduce
    Expected vs actual behavior
    Python version and OS
    Relevant code snippets or error messages


### Feature Requests

When proposing features:

    Describe the use case
    Explain why it would be valuable
    Suggest potential implementation approach
    Consider backward compatibility


### Code Contributions

We welcome:

    Bug fixes
    New emulator implementations
    New inference methods
    Interface modules for additional libraries
    Performance improvements
    Documentation improvements
    Test coverage improvements


## Development Tips

### Running Specific Tests

```bash
# Run tests matching a pattern
pytest -k "test_parameter"

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Checking Coverage

```bash
# Generate HTML coverage report
pytest --cov=c2i2o --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Using Pre-commit Hooks

Pre-commit hooks automatically format and check your code:

```bash

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

## Release Process

Releases are handled by maintainers. The process is documented in [RELEASING.md](RELEASING.md).

For contributors:

- Releases follow [Semantic Versioning](https://semver.org/)
- CHANGELOG.md tracks all changes
- GitHub Actions automate PyPI publishing
- Each release is tagged and has GitHub release notes

To see when your contribution will be released, check:
- [Milestones](https://github.com/KIPAC/c2i2o/milestones) for planned releases
- [CHANGELOG.md](CHANGELOG.md) Unreleased section for pending changes


## Questions?

If you have questions, feel free to:

    Open an issue for discussion
    Contact the maintainer: Eric Charles (echarles@stanford.edu)
    Check existing issues and pull requests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
