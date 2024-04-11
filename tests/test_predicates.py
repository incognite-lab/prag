import numpy as np
import pytest
from testing_utils import Apple

from rddl.predicate import Near


# @pytest.fixture
def get_me_dem_apples():
    apple1 = Apple("apple1")
    apple2 = Apple("apple2")
    return apple1, apple2


def test_predicate_near(get_me_dem_apples):
    apple1, apple2 = get_me_dem_apples
    apples_dict = {"object_A": apple1, "object_B": apple2}

    near_bare = Near()

    near_reused = Near(object_A=near_bare.get_variable("object_A"), object_B=near_bare.get_variable("object_B"))

    near_bare.bind(apples_dict)
    near_reused.bind(apples_dict)

    assert near_bare() == near_reused()
    assert near_reused.object_A() is apple1
    assert near_reused.object_B() is apple2
    assert near_bare.object_A == near_reused.object_A
    assert near_bare.object_B == near_reused.object_B

    apple1.set_position(np.array([0, 0, 0]))
    apple2.set_position(np.array([0, 0, 0]))

    assert near_bare() and near_reused(), "Both near predicates should be true"

    apple1.set_position(np.array([0, 0, 1]))

    assert not (near_bare() or near_reused()), "Both near predicates should be false"


if __name__ == "__main__":
    test_predicate_near(get_me_dem_apples())
