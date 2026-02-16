import random
from collections import deque, namedtuple


# --------------------------------------------------
# Transition Tuple
# --------------------------------------------------

# Each experience = (state, action, reward, next_state, done)
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done")
)


# --------------------------------------------------
# Replay Memory
# --------------------------------------------------

class ReplayMemory:
    """
    Experience Replay Buffer

    Stores past experiences and samples them randomly
    to stabilize training.
    """

    def __init__(self, capacity=100000):
        """
        capacity: maximum number of stored transitions
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in memory
        """
        self.memory.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns current size of memory
        """
        return len(self.memory)
