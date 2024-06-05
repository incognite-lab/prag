from testing_utils import TiagoGripper, Apple
from rddl.rddl_sampler import RDDLWorld
from rddl.entities import Gripper


if __name__ == "__main__":
    rddl_world = RDDLWorld()

    gen = rddl_world.sample_generator(5)

    for _ in range(5):
        action = next(gen)

        for v in rddl_world.get_created_variables():
            if v.is_bound():
                continue  # skip already bound variables
            if issubclass(v.type, Gripper):
                v.bind(TiagoGripper())
            else:
                v.bind(Apple())

        print(f"Current action: {action}\nCurrent reward: {action.compute_reward()}")
