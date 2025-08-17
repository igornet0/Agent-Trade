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
            # собираем по позициям произвольной длины сэмплы
            positional_batches: list[list] | None = None
            any_data = False
            for buf in buffers:
                try:
                    sample = next(buf)
                    any_data = True
                    if not isinstance(sample, (list, tuple)):
                        sample = (sample,)
                    if positional_batches is None:
                        positional_batches = [[] for _ in range(len(sample))]
                    # выравниваем длину
                    if len(sample) > len(positional_batches):
                        positional_batches.extend([[] for _ in range(len(sample) - len(positional_batches))])
                    for idx, val in enumerate(sample):
                        positional_batches[idx].append(val)
                except StopIteration:
                    continue
            if not any_data or not positional_batches:
                break

            batched_outputs = []
            for items in positional_batches:
                if not items:
                    batched_outputs.append(None)
                    continue
                # попытаться стекнуть как numpy, иначе пропустить
                try:
                    tensor = torch.as_tensor(np.stack(items), dtype=torch.float32)
                except Exception:
                    try:
                        tensor = torch.as_tensor(items, dtype=torch.float32)
                    except Exception:
                        tensor = None
                batched_outputs.append(tensor)

            yield tuple(batched_outputs)

class TimeSeriesTransform(IterableDataset):
    def __init__(self, loaders: list[LoaderTimeLine], agent: Agent, batch_size: int = 32, mixed: bool = True):
        self.generator = BatchGenerator(loaders, agent, batch_size, mixed)

    def __len__(self):
        return 100

    def __iter__(self):
        return iter(self.generator)