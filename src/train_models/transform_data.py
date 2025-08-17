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
                        # Ensure arrays
                        s_x = np.asarray(s_x)
                        s_y = np.asarray(s_y)
                        s_t = np.asarray(s_t)
                        s_e1, s_e2 = s_t, None
                    elif len(sample) == 4:
                        # (x, y, x_pred, time) в нашей create_time_line_loader порядок (x, y, x_pred, time)
                        s_x, s_y, s_e1, s_e2 = sample
                        s_x = np.asarray(s_x)
                        s_y = np.asarray(s_y)
                        s_e1 = np.asarray(s_e1)
                        s_e2 = np.asarray(s_e2)
                    else:
                        continue
                    # Сжимаем 1D для y (pred_len,), оставляя 2D где нужно
                    sx = np.asarray(s_x)
                    sy = np.asarray(s_y)
                    se1 = None if s_e1 is None else np.asarray(s_e1)
                    se2 = None if s_e2 is None else np.asarray(s_e2)

                    # y может быть Series/DataFrame — приведём к 1D
                    if sy.ndim > 1 and sy.shape[0] == 1:
                        sy = sy.reshape(-1)
                    if sy.ndim == 2 and sy.shape[1] == 1:
                        sy = sy[:, 0]

                    x_list.append(sx)
                    y_list.append(sy)
                    if se1 is not None:
                        extra1_list.append(se1)
                    if se2 is not None:
                        extra2_list.append(se2)

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