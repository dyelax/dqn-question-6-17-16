#Saving/OpenAI Gym settings
MODEL_SAVE_DIR = 'save/model/'
MONITOR_PATH = 'save/monitor/dqn0'
API_KEY = 'sk_cTl9FELSxaodQvkSAzPHg'

#Data
FRAME_HEIGHT = FRAME_WIDTH = 84

#Implementation constants
SKIP_FRAMES = 4 #Num frames to skip (and repeat action) between each observation
UPDATE_FREQ = 4 #Num actions to take between each SGD update
REPLAY_SIZE = 1e5 #Number of elements in experience replay data (Paper uses 1e6)
REPLAY_START_SIZE = 5e4 #Don't learn (use random policy) until this many iters
DISCOUNT = 0.99 #Discount factor for q learning
NOOP_MAX = 30 #Max num "do nothing" actions performed by agent at start of episode
TRAIN_FRAMES = 1e7 #The total number of frames to train on (Paper uses 5e7)
SAVE_FREQ = 1e6 #Num frames between each save of the model
TARGET_UPDATE_FREQ = 1e4 #Num frames between each update of the target network

#Epsilon annealing
EPS_START = 1
EPS_END = 0.1
EPS_ANNEAL_PERIOD = 1e6
EPS_DIFF = (EPS_START - EPS_END) / EPS_ANNEAL_PERIOD

#Hyperparameters
BATCH_SIZE = 32
HIST_LEN = 4 #Num most recent frames used as input
LEARN_RATE = 2.5e-4
MOMENTUM = 0.95