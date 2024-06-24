from collections import deque
import pandas as pd


action_set = {
    "Approach",
    "Grasp",
    "Drop",
    "Withdraw",
    "Move",
    "Rotate",
    "Transform",
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

sequence_length = 5
RES_QUE = deque()


def explore(current_action, depth, sequence):
    if depth == sequence_length:
        RES_QUE.append(sequence + [current_action])
        return 1
    else:
        return sum([explore(action, depth + 1, sequence + [current_action]) for action in transitions[current_action]])


def get_sequence_from_csv(filename) -> list[list[str]]:
    new_df = pd.read_csv(filename, index_col=0)
    return new_df.to_numpy().tolist()


if __name__ == "__main__":
    total_sequences = 0
    for start_action in starting_action_set:
        number_of_sequences = explore(start_action, 1, [])
        print(f"Found {number_of_sequences} sequences for {start_action} as starting action.")
        total_sequences += number_of_sequences

    print(f"Total number of sequences: {total_sequences}")
    # print(RES_QUE)  # this will print all the sequences; might be very long!

    # Store sequences in csv
    df = pd.DataFrame(RES_QUE)
    df.to_csv(f"sequences-len_{sequence_length}.csv")

    # Load sequences from csv
    # nsq = get_sequence_from_csv(f"sequences-len_{sequence_length}.csv")
    # print(nsq)
