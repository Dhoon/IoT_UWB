import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code_review/ (this file: code_review/src/)

# Data: one file per location, <DATA_DIR>/<location>.json (first 80 % of the sessions of each location)
DATA_DIR = os.path.join(ROOT, 'data', 'train')
TEST_DIR = os.path.join(os.path.dirname(DATA_DIR), 'test')   # last 20 % (test.py)
LOCATIONS = list(range(1, 13))

ANCHOR_POSITIONS = {
    1: (-0.75, 1.50),
    2: ( 0.75, 1.50),
    3: (-0.75, 0.00),
    4: ( 0.75, 0.00),
    5: (-0.57, -0.30),
    6: ( 0.57, -0.30),
}
ANCHORS = list(ANCHOR_POSITIONS)

C = 299702547
DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)
UINT32 = 1 << 32

# Multi-seed training: one independent run per seed (train.py / test.py --seeds)
SEEDS = [0, 1, 2, 3, 4]
MODEL_DIR = os.path.join(ROOT, 'models')

# DQN
BATCH_SIZE = 64
GAMMA = 0.9
TAU = 0.005
MEMORY_SIZE = 3000
MAX_STEP = 50

# Stage 1
EPISODES = 500
ANCHOR_DROP_PROB = 0.0
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
LR = 1e-4

# Stage 2 (fine-tuning)
FINETUNE_EPISODES = 300
FINETUNE_DROP_PROB = 0.2
FINETUNE_EPS_START = 0.3
FINETUNE_LR = 5e-5

# Reward
BETA = 5.0              # azimuth error scale (deg)
GEOM_EPS = 0.3          # R_geom blind zone: |q| > 1 - GEOM_EPS / baseline (m) is penalized
