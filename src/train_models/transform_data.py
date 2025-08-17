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
        while True:
            any_data = False
            x_list, y_list, extra1_list, extra2_list = [], [], [], []
            for buf in buffers:
                try:
                    window_df = next(buf)  # pd.DataFrame окна таймсерии
                    any_data = True
                except StopIteration:
                    continue

                # Преобразуем окно в тензоры для конкретного агента
                try:
                    processed = agent.preprocess_data_for_model(window_df.copy())
                except Exception:
                    continue

                # Нормализуем в кортеж (x, y, extra1, extra2)
                sample_tuple = None
                if isinstance(processed, (list, tuple)):
                    if len(processed) == 3:
                        # (x, y, time)
                        x_np, y_np, t_np = processed
                        sample_tuple = (x_np, y_np, t_np, None)
                    elif len(processed) == 4:
                        # AgentTradeTime.train -> (x, x_pred, y, time) -> переставим
                        x_np, x_pred_np, y_np, t_np = processed
                        sample_tuple = (x_np, y_np, x_pred_np, t_np)
                    else:
                        # неизвестный формат
                        continue
                else:
                    # неожиданный тип
                    continue

                x_np, y_np, e1_np, e2_np = sample_tuple
                try:
                    x_list.append(np.asarray(x_np))
                    y_list.append(np.asarray(y_np))
                    if e1_np is not None:
                        extra1_list.append(np.asarray(e1_np))
                    if e2_np is not None:
                        extra2_list.append(np.asarray(e2_np))
                except Exception:
                    continue

            if not any_data or not x_list:
                break

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