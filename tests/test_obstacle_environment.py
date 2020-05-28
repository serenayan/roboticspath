import pytest
import numpy as np
import matplotlib.pyplot as plt

from obstacle_environment import START_ENV


# usually skip plotting tests
@pytest.mark.skip(reason="Uncomment this line to verify plotting works")
def test_obstacle_environment_plotting():
    START_ENV.plot()

@pytest.mark.skip(reason="Uncomment this line to verify plotting works")
def test_plot_waypoints():
    test = np.array(
        [[30, 50], [30, 50, 40, 30], [70, 80], [70, 80, -10, 7]])
    START_ENV.plot_path(test)
    plt.show()

def test_cvx_ineqs():
    # TODO: test A,b matrix generation
    assert True
