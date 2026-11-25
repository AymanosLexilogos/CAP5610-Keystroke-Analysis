import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import utils as U  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return U.load_data()  # raises with download instructions if the CSV is missing


@pytest.fixture(scope="session")
def X31(df):
    return U.make_features(df, "raw31")[0]


@pytest.fixture(scope="session")
def X21(df):
    return U.make_features(df, "log21")[0]
