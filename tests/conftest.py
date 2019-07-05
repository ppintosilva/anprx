import pytest

def pytest_addoption(parser):
    parser.addoption("--plot", action="store", default=0)

@pytest.fixture
def plot(request):
    return bool(int(request.config.getoption("--plot")))
