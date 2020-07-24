#TODO: move all parameters to this file

# Program parameters
save_dir = './tmp/'

# Training parameters
EPSILON_MIN = 0
EPSILON_MAX = 0
EPSILON_DECAY = 0.00075
MEMORY_CAPACITY = 50000
TARGET_UPDATE = 300
SIZE_HIDDEN = 16
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0001
MAX_STEPS = 2000
ACTIVATION = 'tanh'
LEARNING_START = 100
N_EPISODES = 20

# Environment parameters
control_style_ego = 'RL'
control_style_other = 'RL'
# control_style_other = 'pre-trained'
train_level = 1
time_interval = 0.5 # seconds
MAX_TIME = 7  # seconds
state_size = 4
action_size = 5

# Car parameters
CAR_LENGTH = 4.0  # m
CAR_WIDTH = 2.0  # m