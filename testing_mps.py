#!/Users/charlie/miniconda3/bin/python3

import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else: 
    print("MPS is not available")