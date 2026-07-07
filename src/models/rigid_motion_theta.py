import torch
import torch.nn as nn
from torch_cubic_spline_grids import CubicBSplineGrid1d


def extract_ts(obj: torch.Tensor,
               n_extracts: int) -> torch.Tensor:
    n = obj.shape[0]
    steps = n // n_extracts
    indexes = torch.torch.arange(0, n, steps).to(obj.device)
    return obj[indexes]

class RigidMotionTheta(nn.Module):

    def __init__(self, n_control_points: int = 10,
                       device: str = 'cuda',
                       n_base_angles: int = None,
                       n_angles: int = None) -> None:
        
        super(RigidMotionTheta, self).__init__()
        
        self.tx_splines = CubicBSplineGrid1d(resolution=n_control_points)
        self.ty_splines = CubicBSplineGrid1d(resolution=n_control_points)
        self.tz_splines = CubicBSplineGrid1d(resolution=n_control_points)
        self.rx_splines = CubicBSplineGrid1d(resolution=n_control_points)
        self.ry_splines = CubicBSplineGrid1d(resolution=n_control_points)
        self.rz_splines = CubicBSplineGrid1d(resolution=n_control_points)

        self.n_base_angles = n_base_angles
        self.n_angles = n_angles
        self.device = device
        self.to(device)


    def set_control_points(self, control_points: torch.Tensor):
        
        self.tx_splines.data = control_points[0:1]
        self.ty_splines.data = control_points[1:2]
        self.tz_splines.data = control_points[2:3]
        self.rx_splines.data = control_points[3:4]
        self.ry_splines.data = control_points[4:5]
        self.rz_splines.data = control_points[5:6]

    def get_control_points(self) -> torch.Tensor:

        tx_cp = self.tx_splines.data
        ty_cp = self.ty_splines.data
        tz_cp = self.tz_splines.data
        rx_cp = self.rx_splines.data
        ry_cp = self.ry_splines.data
        rz_cp = self.rz_splines.data

        cps = torch.cat([tx_cp, ty_cp, tz_cp, rx_cp, ry_cp, rz_cp])

        return cps

    def forward(self, timesteps: torch.Tensor,
                      sub_sample: bool = False) -> torch.Tensor:
        
        if sub_sample:
            timesteps = torch.linspace(0, 1, self.n_base_angles).unsqueeze(0).to(self.device)
            tx = self.tx_splines(timesteps)
            ty = self.ty_splines(timesteps)
            tz = self.tz_splines(timesteps)
            rx = self.rx_splines(timesteps)
            ry = self.ry_splines(timesteps)
            rz = self.rz_splines(timesteps)
            
            params = torch.cat([tx, ty, tz, rx, ry, rz]).T
            params = extract_ts(params, self.n_angles)
        else:
            timesteps = timesteps / timesteps.max()
            tx = self.tx_splines(timesteps)
            ty = self.ty_splines(timesteps)
            tz = self.tz_splines(timesteps)
            rx = self.rx_splines(timesteps)
            ry = self.ry_splines(timesteps)
            rz = self.rz_splines(timesteps)    

            params = torch.cat([tx, ty, tz, rx, ry, rz]).T

        del tx, ty, tz, rx, ry, rz, timesteps
        
        return params


