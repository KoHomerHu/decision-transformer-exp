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

    model = DecisionTransformer(
        state_dim, 
        action_dim, 
        max_traj_len=max_traj_len
    )

    dataset = TrajectoryDataset(
        action_dim, 
        "behavioural_trajectory_data.pkl", 
        max_traj_len=max_traj_len
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    num_epochs = 100
    iterator = iter(cycle(dataloader))
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            for _ in range(10):
                batch = next(iterator)
                rtg, obs, act = batch['rtg'].float(), batch['state'].float(), batch['action'].float()

                pred_act = torch.zeros_like(act)
                memory = None

                for i in range(0, max_traj_len):
                    pred_action, memory = model(rtg[:,i,:], obs[:,i,:], memory)
                    pred_act[:, i, :] += pred_action.squeeze(1)

                loss = criterion(pred_act, act)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.update(1)
            pbar.set_description("Epoch {} Loss: {:.4f}".format(epoch, loss.item()))

    
    # Create models folder if not existing
    if not os.path.exists("./models"):
        os.makedirs("./models")

    torch.save(model.state_dict(), "./models/SentryGPT-beta.pt")

