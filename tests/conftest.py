import pytest

import seaduck as sd

from . import utils


@pytest.fixture(scope="session")
def ds(request):
    return utils.get_dataset(request.param)


@pytest.fixture(scope="session")
def od(request):
    return sd.OceData(utils.get_dataset(request.param))


@pytest.fixture(scope="session")
def tp(request):
    return sd.topology(utils.get_dataset(request.param))
