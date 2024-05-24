from rddl import Entity, AtomicAction, LogicalOperand


class RDDLTask:

    def __init__(self, actions: list[AtomicAction], objects: list[Entity], initial_state: LogicalOperand, goal_state: LogicalOperand) -> None:
        self._actions = actions
        self._objects = objects
        self._initial_state = initial_state
        self._goal_state = goal_state

    def current_action(self) -> AtomicAction:
        raise NotImplementedError

    def next_action(self) -> AtomicAction:
        raise NotImplementedError

    def current_reward(self) -> float:
        raise NotImplementedError
