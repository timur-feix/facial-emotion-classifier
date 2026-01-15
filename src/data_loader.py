import torch
from random import shuffle

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle_=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle_
    
    def __iter__(self):
        indices = list(range(0, len(self.dataset)))
        if self.shuffle:
            shuffle(indices)
        
        for s in range(0, len(indices), self.batch_size):
            batch_indices = indices[s:s + self.batch_size]

            batch = [self.dataset[i] for i in batch_indices]
            xs, ys = zip(*batch)

            yield torch.stack(xs), torch.stack(ys)


    