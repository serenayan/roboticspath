import pytest

from obstacle_environment import START_ENV


# usually skip plotting tests
@pytest.skip
def test_obstacle_environment_plotting():
    START_ENV.plot()


def test_cvx_ineqs():
    # TODO: test A,b matrix generation
    assert True
