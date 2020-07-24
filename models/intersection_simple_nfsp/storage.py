import itertools
import math
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # # Uncomment if to use importance sampling
        # n_samples = min(batch_size, len(self.buffer))
        # samples = random.sample(self.buffer, int(n_samples/2))
        # recent_samples = np.array(self.buffer)[-n_samples:]
        # recent_rewards = abs(recent_samples[:, 2])  # larger are worse cases

        # if sum(recent_rewards) > 0:
        #     p = recent_rewards/sum(recent_rewards)
        #     p = np.array(p.tolist())
        #     importance_samples_id = np.random.choice(range(p.size), int(n_samples / 2), replace=True, p=p)
        #     importance_samples = recent_samples[importance_samples_id]
        #     state, action, reward, next_state, done = zip(*(samples + importance_samples.tolist()))
        # else:
        #     state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        # No importance sampling; comment line 36 and uncomment above to enable importance sampling
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class ReservoirBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        state = np.expand_dims(state, 0)
        self.buffer.append((state, action))
    
    def sample(self, batch_size):
        # Efficient Reservoir Sampling
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        n = len(self.buffer)
        reservoir = list(itertools.islice(self.buffer, 0, batch_size))
        threshold = batch_size * 4
        idx = batch_size
        while (idx < n and idx <= threshold):
            m = random.randint(0, idx)
            if m < batch_size:
                reservoir[m] = self.buffer[idx]
            idx += 1
        
        while (idx < n):
            p = float(batch_size) / idx
            u = random.random()
            g = math.floor(math.log(u) / math.log(1 - p))
            idx = idx + g
            if idx < n:
                k = random.randint(0, batch_size - 1)
                reservoir[k] = self.buffer[idx]
            idx += 1
        state, action = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action
    
    def __len__(self):
        return len(self.buffer)