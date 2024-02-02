import torch
from decision_transformer import *
from utils import *
from tqdm import tqdm
import os


if __name__ == '__main__':
    state_dim = 5
    action_dim = 7
    max_traj_len = 50
    batch_size = 256
    num_iterations = 5000
    warmup_steps = 200
    label_smoothing = 0.1
    gamma = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecisionTransformer(
        state_dim, 
        action_dim, 
        max_traj_len=max_traj_len, 
        warmup_steps= warmup_steps
    ).to(device)

    dataset = TrajectoryDataset(
        action_dim, 
        "behavioural_trajectory_data.pkl", 
        max_traj_len=max_traj_len
    )

    try:
        model.load_state_dict(torch.load("./models/SentryGPT-beta.pt"))
    except:
        pass

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    dataloader = InfiniteSampler(state_dim, action_dim, dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1e-1 * model.d_model ** -0.5)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=model.warmup)

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

            targ_act = act[:,-1,:]
            pred_act = model(rtg, obs, act[:,:-1,:])
            # print("\n", pred_act[-1])
            log_probs = torch.log(pred_act + 1e-9)
            loss = -torch.sum(targ_act * torch.pow((1 - pred_act), gamma) * log_probs, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch += 1
            pbar.update(1)
            pbar.set_description("Iteration {} Loss: {:.4f}".format(epoch, loss.item()))

            loss_lst.append(loss.item())

    
    # Create models folder if not existing
    if not os.path.exists("./models"):
        os.makedirs("./models")

    torch.save(model.state_dict(), "./models/SentryGPT-beta.pt")

    # use pyplot to plot the loss curve
    import matplotlib.pyplot as plt
    plt.plot(loss_lst)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.savefig("./loss_curve.png")
