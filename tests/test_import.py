"""Test function that import libraries"""


def test_import_c2i2o() -> None:
    """Test that we can import c2i2o"""
    import c2i2o  # pylint: disable=import-outside-toplevel

    assert c2i2o.__version__
