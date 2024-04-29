import numpy as np
import pytest
import testing_utils

from rddl import Entity, Variable
from rddl.entities import Gripper, Location, ObjectEntity
from rddl.rddl_sampler import RDDLWorld


def test_world():
    rddl_world = RDDLWorld()
    actions, variables = rddl_world.sample_world()

    str_actions = '\n\t'.join([repr(a) for a in actions])
    print(f"Actions:\n\t{str_actions}")
    str_variables = '\n\t'.join([repr(v) for v in variables])
    print(f"Variables:\n\t{str_variables}")


if __name__ == "__main__":
    test_world()
