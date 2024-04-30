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


def test_world_generator():
    rddl_world = RDDLWorld()

    n_samples = 10
    actions = []
    gen = rddl_world.sample_generator(n_samples)

    for _ in range(n_samples):
        action = next(gen)
        print(f"Generated action: {action}")
        actions.append(action)

    variables = rddl_world.get_created_variables()

    str_actions = '\n\t'.join([repr(a) for a in actions])
    print(f"Actions:\n\t{str_actions}")
    str_variables = '\n\t'.join([repr(v) for v in variables])
    print(f"Variables:\n\t{str_variables}")


def test_world_generator_random_reject():
    rddl_world = RDDLWorld()

    n_samples = 100
    actions = []
    gen = rddl_world.sample_generator(n_samples)

    for _ in range(n_samples):
        action = next(gen)
        if np.random.rand() < 0.5:
            action = gen.send(False)
            print(f"Rejected action: {action}, regenerating...")
        else:
            print(f"Generated action: {action}")
            actions.append(action)

    variables = rddl_world.get_created_variables()

    str_actions = '\n\t'.join([repr(a) for a in actions])
    print(f"Actions:\n\t{str_actions}")
    str_variables = '\n\t'.join([repr(v) for v in variables])
    print(f"Variables:\n\t{str_variables}")


if __name__ == "__main__":
    # test_world()
    test_world_generator()
    # test_world_generator_random_reject()
