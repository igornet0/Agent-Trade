import numpy as np
import torch
import pandas as pd
from torch.utils.data import IterableDataset

from Dataset import LoaderTimeLine
from MMM import Agent

class BatchGenerator:
    def __init__(self, loaders: list[LoaderTimeLine], agent: Agent, batch_size: int = 32, mixed: bool = True):
        self.loaders = loaders
        self.agent = agent
        self.batch_size = batch_size
        self.mixed = mixed

    def __iter__(self):
        return self.generate_batches()

    def generate_batches(self):
        buffers = [iter(loader) for loader in self.loaders]
        while True:
            batch_x = []
            batch_y = []
            batch_time = []
            for buf in buffers:
                try:
                    sample = next(buf)
                    batch_x.append(sample[0])
                    batch_y.append(sample[1])
                    if len(sample) > 2 and sample[2] is not None:
                        batch_time.append(sample[2])
                except StopIteration:
                    continue
            if not batch_x:
                break
            x = torch.as_tensor(np.stack(batch_x), dtype=torch.float32)
            y = torch.as_tensor(np.stack(batch_y), dtype=torch.float32)
            if batch_time:
                t = torch.as_tensor(np.stack(batch_time), dtype=torch.float32)
            else:
                t = None
            yield (x, y, t)

class TimeSeriesTransform(IterableDataset):
    def __init__(self, loaders: list[LoaderTimeLine], agent: Agent, batch_size: int = 32, mixed: bool = True):
        self.generator = BatchGenerator(loaders, agent, batch_size, mixed)

    def __len__(self):
        return 100

    def __iter__(self):
        return iter(self.generator)