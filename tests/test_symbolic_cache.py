import os

import numpy as np
import pytest

import rddl

# os.environ["PYTHONPATH"] = os.getcwd()
os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "tests")
from testing_utils import Apple

from rddl.core import Operand

None
from rddl.operator import NotOp
from rddl.predicate import Near


# @pytest.fixture
def get_me_dem_apples():
    apple1 = Apple("apple1")
    apple2 = Apple("apple2")
    return apple1, apple2


def test_predicate_near():
    Operand.set_cache_symbolic()
    near_bare = Near()

    near_reused = Near(object_A=near_bare.get_variable("object_A"), object_B=near_bare.get_variable("object_B"))


def test_compund_predicate_near():
    Operand.set_cache_symbolic()
    near = Near()
    assert not near.decide()

    compund = NotOp(near)
    assert not compund.decide(), "Compound predicate 'Not(Near)' should be false"

    double_compund = NotOp(compund)
    assert not double_compund.decide(), "Compound predicate 'Not(Not(Near))' should be true"


if __name__ == "__main__":
    test_predicate_near()
    test_compund_predicate_near()
