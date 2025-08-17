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
        agent = self.agent
        seq_len = agent.model_parameters.get("seq_len", 50)
        pred_len = agent.model_parameters.get("pred_len", 5)

        x_list, y_list, extra1_list, extra2_list = [], [], [], []

        while True:
            any_data = False
            for buf in buffers:
                try:
                    window_df = next(buf)
                    any_data = True
                except StopIteration:
                    continue

                # На основе окна генерируем выборки нужной формы через агент
                try:
                    samples_iter = agent.create_time_line_loader(window_df, pred_len, seq_len)
                except Exception:
                    continue

                for sample in samples_iter:
                    if not isinstance(sample, (list, tuple)):
                        continue
                    if len(sample) == 3:
                        # (x, y, time)
                        s_x, s_y, s_t = sample
                        s_e1, s_e2 = s_t, None
                    elif len(sample) == 4:
                        # (x, y, x_pred, time) в нашей create_time_line_loader порядок (x, y, x_pred, time)
                        s_x, s_y, s_e1, s_e2 = sample
                    else:
                        continue
                    try:
                        x_list.append(np.asarray(s_x))
                        y_list.append(np.asarray(s_y))
                        if s_e1 is not None:
                            extra1_list.append(np.asarray(s_e1))
                        if s_e2 is not None:
                            extra2_list.append(np.asarray(s_e2))
                    except Exception:
                        continue

                    if len(x_list) >= self.batch_size:
                        x = torch.as_tensor(np.stack(x_list[: self.batch_size]), dtype=torch.float32)
                        y = torch.as_tensor(np.stack(y_list[: self.batch_size]), dtype=torch.float32)
                        e1 = torch.as_tensor(np.stack(extra1_list[: self.batch_size]), dtype=torch.float32) if len(extra1_list) >= self.batch_size else None
                        e2 = torch.as_tensor(np.stack(extra2_list[: self.batch_size]), dtype=torch.float32) if len(extra2_list) >= self.batch_size else None
                        yield (x, y, e1, e2)
                        # remove used
                        x_list = x_list[self.batch_size:]
                        y_list = y_list[self.batch_size:]
                        extra1_list = extra1_list[self.batch_size:] if e1 is not None else []
                        extra2_list = extra2_list[self.batch_size:] if e2 is not None else []

            if not any_data:
                break

        # остаток
        if x_list:
            x = torch.as_tensor(np.stack(x_list), dtype=torch.float32)
            y = torch.as_tensor(np.stack(y_list), dtype=torch.float32)
            e1 = torch.as_tensor(np.stack(extra1_list), dtype=torch.float32) if extra1_list else None
            e2 = torch.as_tensor(np.stack(extra2_list), dtype=torch.float32) if extra2_list else None
            yield (x, y, e1, e2)

class TimeSeriesTransform(IterableDataset):
    def __init__(self, loaders: list[LoaderTimeLine], agent: Agent, batch_size: int = 32, mixed: bool = True):
        self.generator = BatchGenerator(loaders, agent, batch_size, mixed)

    def __len__(self):
        return 100

    def __iter__(self):
        return iter(self.generator)