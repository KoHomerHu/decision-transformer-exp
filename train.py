import torch
from decision_transformer import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import os


if __name__ == '__main__':
    state_dim = 5
    action_dim = 7
    max_traj_len = 50
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecisionTransformer(
        state_dim, 
        action_dim, 
        max_traj_len=max_traj_len
    ).to(device)

    dataset = TrajectoryDataset(
        action_dim, 
        "behavioural_trajectory_data.pkl", 
        max_traj_len=max_traj_len
    )

    # try:
    #     model.load_state_dict(torch.load("./models/SentryGPT-beta.pt"))
    # except:
    #     pass

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    dataloader = InfiniteSampler(state_dim, action_dim, dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    num_epochs = 100
    iterator = iter(cycle(dataloader))

    with tqdm(total=num_epochs) as pbar:
        epoch = 0
        while epoch < num_epochs:

            batch = next(iterator)
            rtg, obs, act = batch['rtg'].float().to(device), batch['state'].float().to(device), batch['action'].float().to(device)
            if act.shape[0] != batch_size:
                continue

            pred_act = torch.zeros_like(act).to(device)
            memory = None

            for i in range(0, max_traj_len):
                pred_action, memory = model(rtg[:,i,:], obs[:,i,:], memory)
                pred_act[:, i, :] += pred_action.squeeze(1)

            loss = criterion(pred_act.reshape((max_traj_len * batch_size, action_dim)), 
                             act.reshape((max_traj_len * batch_size, action_dim)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch += 1
            pbar.update(1)
            pbar.set_description("Epoch {} Loss: {:.4f}".format(epoch, loss.item()))

    
    # Create models folder if not existing
    if not os.path.exists("./models"):
        os.makedirs("./models")

    torch.save(model.state_dict(), "./models/SentryGPT-beta.pt")

