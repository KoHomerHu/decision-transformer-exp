import torch
import torch.nn.functional as F
from decision_transformer import *
from utils import *
from tqdm import tqdm
import os

warmup_steps = 200
def warmup(step_num):
    step_num += 1
    return min(step_num ** -0.5, step_num * warmup_steps ** -1.5)


if __name__ == '__main__':
    state_dim = 5
    action_dim = 7
    max_traj_len = 50
    batch_size = 256
    num_iterations = 5000
    label_smoothing = 0.1
    gamma = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        1 + state_dim + action_dim
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
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    iterator = iter(cycle(dataloader))
    loss_lst = []

    with tqdm(total=num_iterations) as pbar:
        epoch = 0
        while epoch < num_iterations:
            X, y = next(iterator)
            X = X.to(device)
            y = y.to(device)
            prediction = model(X, padding = False).squeeze(1)
            action_prob = F.softmax(prediction[:,-action_dim:], dim=-1)
            # print(action_prob)
            # print(y)
            log_probs = torch.log(action_prob + 1e-9)
            loss = -torch.sum(y * torch.pow((1 - action_prob), gamma) * log_probs, dim=-1).mean()
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
