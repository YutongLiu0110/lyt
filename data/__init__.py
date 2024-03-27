from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple
from torch.utils import data as dt

# 这将 DATA_ROOT 设置为当前文件的父目录的绝对路径。__file__ 是一个特殊变量，表示当前文件的路径。
DATA_ROOT = Path(__file__).parent.resolve()

# 这个函数接受以下参数：feat_dim：表示特征维度的整数。pred_len：表示预测长度的整数。batch_size：表示批大小的整数。函数返回一个包含三个数据加载器对象的元组。


def get_dataloaders(
        syn_type: str, syn_param: int, feat_dim: int, pred_len: int, batch_size: int
) -> Tuple[dt.DataLoader]:
    tgt_train_path, tgt_valid_path = (
        DATA_ROOT / "loadshuju" / "targettrain.csv",
        DATA_ROOT / "loadshuju" / "targettest.csv",
    )
    tgt_trainset, tgt_validset = (
        CSVDataset(tgt_train_path, feat_dim, pred_len),  # 使用CSVDataset加载目标数据
        CSVDataset(tgt_valid_path, feat_dim, pred_len),
    )
    src_path = DATA_ROOT / "loadshuju" / "source1.csv"
    src_dataset = CSVDataset(src_path, feat_dim, pred_len)  # 使用CSVDataset加载源数据

    return (
        dt.DataLoader(src_dataset, batch_size=batch_size, shuffle=True),
        dt.DataLoader(tgt_trainset, batch_size=batch_size, shuffle=True),
        dt.DataLoader(tgt_validset, batch_size=batch_size, shuffle=False),
    )


class CSVDataset(Dataset):
    def __init__(self, file_path, feat_dim, pred_len):
        self.data = pd.read_csv(file_path)  # 读取CSV文件并转换为DataFrame
        self.data = self.data.iloc[:, 1:].values  # 移除时间步，并将数据转换为NumPy数组

        # 归一化处理
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)

        self.feat_dim = feat_dim
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 从CSV数据中提取特征和标签
        features = self.data[index, :self.pred_len]
        targets = self.data[index, self.pred_len:]

        features = features.astype(np.float32)  # 将特征元素类型转换为float32
        targets = targets.astype(np.float32)  # 将元素类型转换为float32
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # 返回特征和标签
        return features, targets
