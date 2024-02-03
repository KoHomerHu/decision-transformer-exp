import torch
import random
import numpy as np
from decision_transformer import *
from env.rmul_env import Env
import torch.nn.functional as F
import torch.distributions as D
import cv2

arena_file = "./env/arena.json"
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

agent = Transformer(state_dim + action_dim + 1).to(device)

agent.load_state_dict(torch.load("./models/SentryGPT-beta2.pt"))
agent.eval()

print(agent)
input("Press Enter to start the evaluation.")

# Create a VideoWriter object to save the video
video_path = "./videos/test.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
x, y = env.arena.get_arena_image().shape[:2]
video_writer = cv2.VideoWriter(video_path, fourcc, 30, (y, x))

reward_to_go = -500 # prompt: expectation of the reward-to-go

for i in range(1):
    state = env.reset()
    state_lst = [state,]
    rtg_lst = [reward_to_go,]
    action_lst = []
    memory = None
    done = truncated = False
    if i == 9:
        print("Recording video...")
    while not (done or truncated):
        state = torch.tensor(state).float().to(device)
        obs = torch.tensor(state_lst).to(device)
        # print(rtg_lst)
        rtg = torch.tensor(rtg_lst).to(device).unsqueeze(-1)
        blank = -2 * torch.ones((1, action_dim)).to(device)
        # print(action_lst)
        act = torch.cat((torch.tensor(action_lst).to(device), blank), dim=-2)
        # print(rtg.shape, obs.shape, act.shape)
        X = torch.cat((rtg, obs, act), dim=-1).unsqueeze(0).float().to(device)
        print(X)
        prediction = agent(X, padding = False).squeeze(1)
        action_prob = F.softmax(prediction[:,-action_dim:], dim=-1)
        print(action_prob)
        action_idx = D.Categorical(action_prob).sample().item()
        print(action_idx)
        print(reward_to_go)
        cnt = 0
        num_skipped_frame = 15
        while cnt <= num_skipped_frame and not (done or truncated):
            next_state, reward, done, _, frame = env.step(action_idx)
            cnt += 1
        state = next_state
        reward_to_go -= reward
        state_lst.append(state)
        # print(rtg)
        rtg_lst.append(reward_to_go)
        action_lst.append(F.one_hot(torch.tensor(action_idx), action_dim).float().tolist())
        state_lst = state_lst[-50:]
        rtg_lst = rtg_lst[-50:]
        action_lst = action_lst[-49:]
        if reward_to_go > 0:
            reward_to_go = -500
            rtg_lst = [reward_to_go,]
            state_lst = [state,]
            action_lst = []

        # Write the frame to the video
        if i == 9:
            video_writer.write(frame)

# Release the video writer and close the video file
video_writer.release()
cv2.destroyAllWindows()