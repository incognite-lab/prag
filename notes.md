# Notes for task generation

A little more in-depth since it apparently should be more general (handling task generation, not just rewards).

## Ideas

### Main idea

Elementary action (reward) vs Complex (Compound?) action (reward).

Elementary actions have methods `calculate` to get the current reward value. Additionally, they have initial and goal predicates (akin to pre-conditions and post-conditions).

### Radical views on actions, tasks & rewards

#### Separability of reward system

Action = reward + some extra functionality (checking execution). Should rewards be handled separately?

#### What is a task?

Task could be very complex (e.g., preparing a meal). Action is still very simple, even when compounded from several elementary actions.

Simple distinction: task is always a strict sequence of operations. Could be parallel but has very clearly separated operations (actions). That is, when an operation is finished, it is assumed to be finished "forever". 



## Issues

### reach - reach (in Poke)

How to interpret? Reach is completed when gripper is near an object. Consequent reach should therefore just "exit" immediately without any action(?)
Maybe a "short push" or short movement should be used instead?

### Push - can it use 'move' element?

Push does not actually work with move, unless move is tailored to pushing.  
(note: reward is easy to define but to generate and evaluate a task is difficult)  
Problem:

```python
move(o, t):  # basic version
  move_gripper_towards(g, t)
```

But what if o is not in the gripper? And does not move with the gripper?

```python
move(o, t):  # ensuring object is held
  if holding(g, o):
    move_gripper_towards(g, t)
```

But what to do when not holding? Try to grasp? Fail?

```python
move(o, t):  # "intelligent" move
  if holding(g, o):
    move_gripper_towards(g, t)
  else:
    while not near(g, o):
      move_gripper_towards(g, o)
    grasp(g, o)
```

But this is kind of cheating - move is no longer elementary action!

```python
move(o, t):  # "assertive" move
  if holding(g, o):
    move_gripper_towards(g, t)
  else:
    fail()
```

This would be correct, but the problem is that this assumes objects can only be moved if they are grasped. And it is assumed objects can be moved if they are grasped (counter-example: large rock).

Possible implementation:

```python
move(o, t):
  move_gripper_towards(g, t)
  if somehow_handle_not_moving_object(o):
    fail()
```

Now both "in-hand" and "out-of-hand" moving works. But how to implement the `somehow_handle_not_moving_object` function? Instant-fail would be incorrect as it might take some time to accelerate heavy objects. Also, temporary stopping to move object (e.g., when pushing) can happen. Therefore, the function would have to handle "delayed movement with hysteresis". This will require several parameters and potentially require keeping history of object states.  
Is that a suitable approach?
