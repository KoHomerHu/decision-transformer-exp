import os
import pickle
from env.rmul_env import Env
from utils import *
import keyboard
import cv2
from time import time

arena_file = "./env/arena.json"
total_robot_num = 1 # first try with only ego robot
robot_configuration = {
    'fov' : 1.22,
    'radius' : 400,
    'arrow_length' : 500
}

env = Env(arena_file, total_robot_num, robot_configuration, 1.0, rendering=True, truncate_size=1000, real_show = False)

filename = "./data/behavioural_trajectory_data.pkl"
data = []
if os.path.exists(filename):
    data = pickle.load(open(filename, 'rb')) # multiple runs of this script will append to the same file
else:
    os.makedirs(os.path.dirname(filename), exist_ok=True)

print("{} trajectories already collected.".format(len(data)))
input("Press Enter to start data collection of 10 trajectories.")

max_traj_len = 50

for i in range(10):
    state = env.reset()
    done = truncated = False
    trajectory = {
        'state' : [],
        'action' : [],
        'reward-to-go' : []
    }
    rewards = []
    start_time = time()
    while not (done or truncated):
        action = 0
        if keyboard.is_pressed('o'):
            action = 6
        elif keyboard.is_pressed('p'):
            action = 5
        if keyboard.is_pressed('w') or keyboard.is_pressed('up'):
            action = 4
        elif keyboard.is_pressed('a') or keyboard.is_pressed('left'):
            action = 1
        elif keyboard.is_pressed('s') or keyboard.is_pressed('down'):
            action = 3
        elif keyboard.is_pressed('d') or keyboard.is_pressed('right'):
            action = 2
        this_time = time()
        next_state = state
        # frame skipping
        cnt = 0
        num_skipped_frame = 15
        while cnt <= num_skipped_frame and not (done or truncated):
            next_state, reward, done, _, _ = env.step(action)
            cnt += 1
        trajectory['state'].append(state)
        rewards.append(reward)
        trajectory['action'].append(action)
        state = next_state
    if len(trajectory['state']) < max_traj_len:
        continue
    trajectory['reward-to-go'] = reward_to_go(rewards)
    data.append(trajectory)
    end_time = time()
    print("FPS = ", len(trajectory['state']) / (end_time - start_time))

print("{} data tuples collected.".format(len(data)))

pickle.dump(data, open(filename, 'wb'))

print("Data collection completed.")

cv2.destroyAllWindows()