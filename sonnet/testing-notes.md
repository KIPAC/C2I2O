# Testing Best Practices for c2i2o

## Floating-Point Comparisons

**Always use `np.isclose()` or `np.allclose()` instead of `==` for floating-point comparisons.**

### Examples:

```python
# ❌ BAD - Can fail due to floating-point precision
assert result == 0.0
assert array1 == array2

# ✅ GOOD - Handles floating-point precision
assert np.isclose(result, 0.0, atol=1e-10)
np.testing.assert_allclose(array1, array2, rtol=1e-5)
```

Choosing Tolerances:

    atol (absolute tolerance): For values near zero
        Use atol=1e-10 for mathematical zeros
        Use atol=1e-6 for physical quantities with known precision

    rtol (relative tolerance): For values far from zero
        Use rtol=1e-5 for general comparisons
        Use rtol=1e-3 for less precise calculations


### NumPy Testing Utilities:

```pyton
import numpy.testing as npt

# For arrays
npt.assert_allclose(actual, expected, rtol=1e-5)
npt.assert_array_equal(actual, expected)  # For exact equality

# For single values
assert np.isclose(actual, expected, atol=1e-10)
```

### Random Number Generation

Always use seeded RNGs for reproducible tests:

```python
# ✅ GOOD - Reproducible
rng = np.random.default_rng(42)
samples = rng.normal(0, 1, 100)

# ❌ BAD - Non-reproducible
samples = np.random.normal(0, 1, 100)
```

### Pytest Fixtures

Use fixtures for common test data:
```python

@pytest.fixture
def cosmology() -> CosmologicalParameters:
    return CosmologicalParameters(omega_m=0.3, ...)

def test_something(cosmology: CosmologicalParameters):
    # Use cosmology fixture
    ...
```


### Testing Exceptions

Use pytest.raises() with match for error messages:

```python
# ✅ GOOD - Checks error type and message
with pytest.raises(ValueError, match="must be positive"):
    validate_positive(-1.0)

# ❌ BAD - Only checks error type
with pytest.raises(ValueError):
    validate_positive(-1.0)
```


### Test Organization

    1. One test class per class being tested
    2. Group related tests together
    3. Use descriptive test names: test_<what>_<condition>_<expected>
    4. Keep tests independent - no shared state


## Common Patterns

### Testing Statistical Results:

```python
def test_mean_is_near_expected(sample_result):
    mean = sample_result.mean()
    # Use generous bounds for statistical tests
    assert abs(mean["param"]) < 0.2  # ~2 sigma for 1000 samples
```

### Testing Save/Load:

```python
def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test"
        obj.save(path)
        loaded_obj = Class.load(path)
        assert obj.config == loaded_obj.config
```

### Testing Reproducibility:

```python
def test_reproducibility():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = function(rng=rng1)
    result2 = function(rng=rng2)

    np.testing.assert_array_equal(result1, result2)
```
