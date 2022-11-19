"""Configuration for pytest."""

# conftest.py

import pytest


def pytest_addoption(parser):
    """Parser adoption.

    Parameters
    ----------
    parser : Any
        configuration for parser adoption
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Configure pytest.

    Parameters
    ----------
    config : any
        configuration for pytest
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Pytest collection.

    Parameters
    ----------
    config : Any
        configuration
    items : Any
        items
    """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
