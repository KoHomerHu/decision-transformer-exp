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
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecisionTransformer(
        state_dim, 
        action_dim, 
        max_traj_len=max_traj_len
    ).to(device)

    dataset = TrajectoryDataset(
        action_dim, 
        "behavioural_trajectory_data_test.pkl", 
        max_traj_len=max_traj_len
    )

    try:
        model.load_state_dict(torch.load("./models/SentryGPT-beta.pt"))
    except:
        pass

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    dataloader = InfiniteSampler(state_dim, action_dim, dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    num_iterations = 1000
    iterator = iter(cycle(dataloader))
    loss_lst = []

    with tqdm(total=num_iterations) as pbar:
        epoch = 0
        while epoch < num_iterations:

            batch = next(iterator)
            rtg, obs, act = batch['rtg'].float().to(device), batch['state'].float().to(device), batch['action'].float().to(device)
            traj_len = rtg.shape[1]
            if act.shape[0] != batch_size:
                continue

            targ_act = act[:,-1,:].float().to(device)
            memory = None

            for i in range(traj_len):
                pred_action, memory = model(rtg[:,i,:], obs[:,i,:], memory)
                act_encoding = F.tanh(model.act_embed(act[:,i,:])).unsqueeze(1)
                memory = torch.cat((memory, act_encoding), dim=-2)
                memory = memory[:, -max_traj_len * 2:, :]
            pred_act = pred_action.squeeze(1).float().to(device)

            # Only update the latter half of the predictions

            loss = criterion(pred_act, targ_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch += 1
            pbar.update(1)
            pbar.set_description("Iteration {} Loss: {:.4f}".format(epoch, loss.item()))

            loss_lst.append(loss.item())

    
    # Create models folder if not existing
    if not os.path.exists("./models"):
        os.makedirs("./models")

    torch.save(model.state_dict(), "./models/SentryGPT-beta2.pt")

    # use pyplot to plot the loss curve
    import matplotlib.pyplot as plt
    plt.plot(loss_lst)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig("./loss_curve.png")
