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
    # rddl_world.set_allowed_entities(
    #     [Gripper, Apple, Tuna]
    # )
    # rddl_world.set_allowed_actions([Approach, Withdraw, Grasp, Drop, Move])
    # rddl_world.set_allowed_predicates([IsReachable])

    n_samples = 5
    # rddl_task = rddl_world.sample_world(n_samples)

    # for o in rddl_task.objects:
    #     my_o = o.instantiate()
    #     o.bind(my_o)

    # reward: float = rddl_task.current_reward()


    actions = []


    gen = rddl_world.sample_generator(n_samples)

    while True:
        try:
            action = next(gen)
        except StopIteration:
            break
        print(f"Generated action: {action}")
        print("World state after action:")
        rddl_world.show_world_state()
        print("")
        actions.append(action)

    variables = rddl_world.get_created_variables()

    str_actions = '\n\t'.join([repr(a) for a in actions])
    print(f"Actions:\n\t{str_actions}")
    str_variables = '\n\t'.join([repr(v) for v in variables])
    print(f"Variables:\n\t{str_variables}")
    print("> Initial state:")
    rddl_world.show_initial_world_state()
    print("> Goal state:")
    rddl_world.show_goal_world_state()


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


def yielder():
    print("function start")
    for i in range(10):
        print("pre-yield")
        data = yield i
        print("post-yield", data)
    print("function end")


if __name__ == "__main__":
    # test_world()
    test_world_generator()
    # # test_world_generator_random_reject()
    # it = 100
    # yd = yielder()
    # print(next(yd))
    # print("Start")
    # while True:
    #     try:
    #         # print("Normal yield")
    #         # out = next(yd)
    #         # print(f"Yield: {out}")
    #         print("Yield from send")
    #         out = yd.send(f"- {it}")
    #         print(f"Yield (from send): {out}")
    #         it += 1
    #     except StopIteration:
    #         print("StopIteration")
    #         break
    # print("Done")
