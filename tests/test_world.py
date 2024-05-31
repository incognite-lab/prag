import numpy as np
import pytest
from tqdm import tqdm
import testing_utils

from rddl import Entity, Variable
from rddl.entities import Gripper, Location, ObjectEntity
from rddl.rddl_sampler import RDDLWorld, Weighter


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

    n_samples = 30
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


def test_sampling_eff():
    rddl_world = RDDLWorld()

    n_sampling_attempts = 10000
    n_samples_per_attempt = 30

    mode_names = {
        Weighter.MODE_INITIAL: "Initial",
        Weighter.MODE_WEIGHT: "Weight",
        Weighter.MODE_SEQUENCE: "Sequence",
        Weighter.MODE_RANDOM: "Random"
    }

    modes = [Weighter.MODE_WEIGHT, Weighter.MODE_INITIAL, Weighter.MODE_SEQUENCE, Weighter.MODE_RANDOM]
    modes += [Weighter.MODE_WEIGHT | Weighter.MODE_INITIAL, Weighter.MODE_WEIGHT | Weighter.MODE_INITIAL | Weighter.MODE_SEQUENCE]
    modes += [Weighter.MODE_WEIGHT | Weighter.MODE_INITIAL | Weighter.MODE_RANDOM, Weighter.MODE_WEIGHT | Weighter.MODE_INITIAL | Weighter.MODE_SEQUENCE | Weighter.MODE_RANDOM]

    for mode in modes:
        rddl_world.reset_weights(mode)
        found_uq_sequences = {}

        for _ in tqdm(range(n_sampling_attempts), position=0, desc="Sampling attempts"):
            gen = rddl_world.sample_generator(n_samples_per_attempt)
            actions = []
            for _ in tqdm(range(n_samples_per_attempt), desc="Samples", leave=False):
                try:
                    action = next(gen)
                except StopIteration:
                    break
                actions.append(action)

            h = tuple(a.__class__.__name__ for a in actions)
            if h in found_uq_sequences:
                found_uq_sequences[h] += 1
            else:
                found_uq_sequences[h] = 1

        print("\n>>>>>>>>>>>>>>>>>>>>")
        print(f"Mode: {mode} [" + ' | '.join([mn for mv, mn in mode_names.items() if mv & mode]) + "]")
        print(f"Total number of unique sequences: {len(found_uq_sequences)} / {n_sampling_attempts} (total attempts, {n_samples_per_attempt} samples per attempt)")
        print(f"Efficiency: {len(found_uq_sequences) / n_sampling_attempts}")
        print(f"Average repeat: {np.mean(list(found_uq_sequences.values()))}")
        print("<<<<<<<<<<<<<<<<<<<<<\n")


if __name__ == "__main__":
    # test_world()
    # test_world_generator()
    test_sampling_eff()
