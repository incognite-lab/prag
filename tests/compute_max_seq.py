from collections import deque


action_set = {
    "Approach",
    "Grasp",
    "Drop",
    "Withdraw",
    "Move",
    "Rotate",
    "Transform"
    "Follow",
}

starting_action_set = {
    "Approach"
}

transitions = {
    "Approach": [
        "Grasp",
        "Withdraw"
    ],
    "Grasp": [
        "Drop",
        "Move",
        "Rotate",
        "Transform",
        "Follow"
    ],
    "Drop": [
        "Grasp",
        "Withdraw"
    ],
    "Withdraw": [
        "Approach",
    ],
    "Move": [
        "Drop",
        "Move",
        "Rotate",
        "Transform",
        "Follow"
    ],
    "Rotate": [
        "Drop",
        "Move",
        "Rotate",
        "Transform",
        "Follow"
    ],
    "Transform": [
        "Drop",
        "Move",
        "Rotate",
        "Transform",
        "Follow"
    ],
    "Follow": [
        "Drop",
        "Move",
        "Rotate",
        "Transform",
        "Follow"
    ]
}

sequence_length = 20
RES_QUE = deque()


def explore(current_action, depth, sequence):
    if depth == sequence_length:
        RES_QUE.append(sequence + [current_action])
        return 1
    else:
        return sum([explore(action, depth + 1, sequence + [current_action]) for action in transitions[current_action]])


if __name__ == "__main__":
    total_sequences = 0
    for start_action in starting_action_set:
        number_of_sequences = explore(start_action, 1, [])
        print(f"Found {number_of_sequences} sequences for {start_action} as starting action.")
        total_sequences += number_of_sequences

    print(f"Total number of sequences: {total_sequences}")
    # print(RES_QUE)  # this will print all the sequences; might be very long!
