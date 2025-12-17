# Release Checklist

Use this checklist when preparing a new release.

## Pre-Release

- [ ] All tests passing (`pytest`)
- [ ] All examples running (`bash examples/run_all_examples.sh`)
- [ ] Pre-commit checks passing (`pre-commit run --all-files`)
- [ ] Documentation builds (`cd docs && make html`)
- [ ] CHANGELOG.md updated
- [ ] Version number updated in:
  - [ ] `pyproject.toml`
  - [ ] `docs/source/conf.py`
- [ ] Dependencies reviewed and updated
- [ ] README.md accurate and up-to-date

## Testing

- [ ] Fresh virtual environment test:
  ```bash
  python -m venv test_env
  source test_env/bin/activate
  pip install -e .
  pytest
  ```

- [ ] Installation from source:
  ```bash
  python -m venv install_test
  source install_test/bin/activate
  pip install .
  python -c "import c2i2o; print(c2i2o.__version__)"
  ```

- [ ] All optional dependencies install:
  ```bash
  pip install .[pytorch]
  pip install .[cosmology]
  pip install .[all]
  ```

- [ ] Documentation builds without errors:
  ```bash
  cd docs
  make clean
  make html
  ```
