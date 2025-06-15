# import open3d
import numpy as np
import torch
from spaghetti.constants import DEBUG
from typing import Tuple, List, Union, Optional
import torch.optim.optimizer
import torch.utils.data

if DEBUG:
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

V_Mesh = Tuple[ARRAY, ARRAY]
T_Mesh = Tuple[T, Optional[T]]
T_Mesh_T = Union[T_Mesh, T]
COLORS = Union[T, ARRAY, Tuple[int, int, int]]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
Optimizer = torch.optim.Adam
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
Subset = torch.utils.data.Subset
