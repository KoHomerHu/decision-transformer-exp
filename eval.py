import torch
import random
import numpy as np
from decision_transformer import *
from env.rmul_env import Env
import os
import cv2

arena_file = "arena.json"
total_robot_num = 1 # first try with only ego robot
state_dim = total_robot_num * 3 + 2
action_dim = 14
robot_configuration = {
    'fov' : 1.22,
    'radius' : 400,
    'arrow_length' : 500
}

total_robot_num = 1 # first try with only ego robot
state_dim = total_robot_num * 3 + 2
action_dim = 7

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env(arena_file, total_robot_num, robot_configuration, 1.0, rendering = True, truncate_size=5000, eval=True)

agent = DecisionTransformer(state_dim, action_dim, max_traj_len=50).to(device)

agent.load_state_dict(torch.load("./models/SentryGPT-beta.pt"))
agent.eval()

# Create a VideoWriter object to save the video
video_path = "./videos/test.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
x, y = env.arena.get_arena_image().shape[:2]
video_writer = cv2.VideoWriter(video_path, fourcc, 30, (y, x))

rtg = torch.Tensor(10000).to(device) # prompt: expectation of the reward-to-go

for i in range(1):
    state = env.reset()
    memory = None
    done = truncated = False
    if i == 9:
        print("Recording video...")
    while not (done or truncated):
        action, memory = agent.take_action(rtg, state, memory)
        next_state, reward, done, truncated, frame = env.step(action)
        state = next_state
        rtg -= reward

        # Write the frame to the video
        if i == 9:
            video_writer.write(frame)

# Release the video writer and close the video file
video_writer.release()
cv2.destroyAllWindows()