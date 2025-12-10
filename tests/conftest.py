import pytest


@pytest.fixture(name="setup_data", scope="package")
def setup_data(request: pytest.FixtureRequest) -> int:
    """Download and setup any needed data"""
    assert request
    ret_val = 0
    return ret_val
