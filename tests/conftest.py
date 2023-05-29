import pytest

import seaduck as sd
from seaduck import utils


@pytest.fixture(scope="session")
def ds(request):
    return utils.get_dataset(request.param)


@pytest.fixture(scope="session")
def od(request):
    return sd.OceData(utils.get_dataset(request.param))


@pytest.fixture(scope="session")
def tp(request):
    return sd.Topology(utils.get_dataset(request.param))
