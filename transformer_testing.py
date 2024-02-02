from decision_transformer import *
import torch
from torch.utils.data import Dataset, Sampler
import random


def inference_test():
    model = Transformer(2)
    model.eval()
    src = torch.rand(5, 1, 2)
    print("src of {} = \n".format(src.shape), src)
    for _ in range(2):
        out = model(src, pred_len=1)
        src = torch.cat((src, out), dim=1)
        print("Untrained Model Prediction: {}\n".format(src.shape), src)


class RepeatDataset(Dataset):
    def __init__(self, feature_dim):
        self.data = []
        self.data.append(torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).unsqueeze(-1))
        self.data.append(torch.tensor([5, 6, 5, 6, 5, 6, 5, 6, 5, 6]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 8, 7, 8, 7, 8, 7, 8, 7, 8]).unsqueeze(-1))
        self.data.append(torch.tensor([9, 10, 9, 10, 9, 10, 9, 10, 9, 10]).unsqueeze(-1))
        self.data.append(torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]).unsqueeze(-1))
        self.data.append(torch.tensor([5, 6, 7, 5, 6, 7, 5, 6, 7, 5]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 8, 9, 7, 8, 9, 7, 8, 9, 7]).unsqueeze(-1))
        self.data.append(torch.tensor([9, 10, 11, 9, 10, 11, 9, 10, 11, 9]).unsqueeze(-1))
        self.data.append(torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2]).unsqueeze(-1))
        self.data.append(torch.tensor([5, 6, 7, 8, 5, 6, 7, 8, 5, 6]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 8, 9, 10, 7, 8, 9, 10, 7, 8]).unsqueeze(-1))
        self.data.append(torch.tensor([9, 10, 11, 12, 9, 10, 11, 12, 9, 10]).unsqueeze(-1))
        self.data.append(torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]).unsqueeze(-1))
        self.data.append(torch.tensor([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 8, 9, 10, 11, 7, 8, 9, 10, 11]).unsqueeze(-1))
        self.data.append(torch.tensor([9, 10, 11, 12, 13, 9, 10, 11, 12, 13]).unsqueeze(-1))
        self.data.append(torch.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4]).unsqueeze(-1))
        self.data.append(torch.tensor([5, 6, 7, 8, 9, 10, 5, 6, 7, 8]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 8, 9, 10, 11, 12, 7, 8, 9, 10]).unsqueeze(-1))
        self.data.append(torch.tensor([1, 4, 1, 4, 1, 4, 1, 4, 1,  4]).unsqueeze(-1))
        self.data.append(torch.tensor([5, 8, 5, 8, 5, 8, 5, 8, 5, 8]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 10, 7, 10, 7, 10, 7, 10, 7, 10]).unsqueeze(-1))
        self.data.append(torch.tensor([1, 7, 2, 1, 7, 2, 1, 7, 2, 1]).unsqueeze(-1))
        self.data.append(torch.tensor([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]).unsqueeze(-1))
        self.data.append(torch.tensor([7, 4, 7, 4, 7, 4, 7, 4, 7, 4]).unsqueeze(-1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
    

class InfiniteSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.length = len(dataset)
        self.batch_size = batch_size
        self.idx = 0
    
    def __iter__(self):
        ret = torch.empty((self.batch_size, 10, 1))
        for i in range(self.batch_size):
            ret[i,:,:] = self.dataset[self.idx]
            self.idx = random.randint(0, self.length - 1)
        yield ret


def training_test():
    dataset = RepeatDataset(1)
    dataloader = InfiniteSampler(dataset, 16)
    iterator = iter(cycle(dataloader))
    
    model = Transformer(1)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=1e-4)
    criterion = torch.nn.MSELoss()
    loss_lst = []

    from tqdm import tqdm
    with tqdm(total=1000) as pbar:
        for i in range(1000):
            batch = next(iterator)
            X, y = batch[:, :-1, :], batch[:, -1, :]
            y_hat = model(X, pred_len=1).squeeze(1)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_lst.append(loss.item())
            pbar.update(1)
            pbar.set_description("Iteration {} Loss: {:.4f}".format(i, loss.item()))

    from matplotlib import pyplot as plt
    plt.plot(loss_lst)
    plt.title("Loss Curve")
    plt.show()

    # trained inference result

    model.eval()
    src = torch.tensor([
        [1, 7, 1, 7, 1, 7, 1, 7, 1],
        [5, 6, 5, 6, 5, 6, 5, 6, 5],
        [1, 2, 3, 1, 2, 3, 1, 2, 3],
        [4, 5, 4, 5, 4, 5, 4, 5, 4],
        [1, 5, 6, 1, 5, 6, 1, 5, 6],
    ]).unsqueeze(-1)

    print("src of {} = \n".format(src.shape), src)
    print("Predicted Sequence:\n")
    out = model(src, pred_len=1)
    print(out)



if __name__ == '__main__':
    # inference_test() # working
    training_test()