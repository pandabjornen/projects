from .models import IKNet_arm
from .models import IKNet
import torch

class Ik_arm:
    def __init__(self, model_path : str):
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


        self.model : torch.Module = IKNet_arm().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def get_angles(self, x, y, z):
        # MPS (Apple Silicon GPU) devices do not support float64 tensors.
        # We conditionally set the dtype to float32 for MPS devices.
        # For other devices (CPU/CUDA), we set dtype to None, which tells
        # PyTorch to infer the default dtype from the input data.
        tensor_dtype = torch.float32 if self.device == "mps" else None
        point = torch.tensor([[x,y,z]], device=self.device, dtype=tensor_dtype)
        with torch.no_grad():
            angles = self.model(point).cpu().squeeze().numpy()        
        return angles[0], angles[1], angles[2]


class Ik_hand:
    def __init__(self, model_path : str):
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


        self.model : torch.Module = IKNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def get_angles(self, x, y, z):
        # MPS (Apple Silicon GPU) devices do not support float64 tensors.
        # We conditionally set the dtype to float32 for MPS devices.
        # For other devices (CPU/CUDA), we set dtype to None, which tells
        # PyTorch to infer the default dtype from the input data.
        tensor_dtype = torch.float32 if self.device == "mps" else None
        point = torch.tensor([[x,y,z]], device=self.device, dtype=tensor_dtype)
        with torch.no_grad():
            angles = self.model(point).cpu().squeeze().numpy()        
        return angles[0], angles[1], angles[2], angles[3]
