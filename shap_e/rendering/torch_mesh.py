from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from .mesh import TriMesh

def srgb_to_linear(channel):
    
    channel[channel <0.0031308] = channel[channel < 0.0031308] *12.92
    channel[channel >= 0.0031308] = 1.055 * (channel[channel >= 0.0031308] ** (1/2.4)) - 0.055
    return channel 

@dataclass
class TorchMesh:
    """
    A 3D triangle mesh with optional data at the vertices and faces.
    """

    # [N x 3] array of vertex coordinates.
    verts: torch.Tensor

    # [M x 3] array of triangles, pointing to indices in verts.
    faces: torch.Tensor

    # Extra data per vertex and face.
    vertex_channels: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)
    face_channels: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)
    
    def tri_mesh(self) -> TriMesh:
        """
        Create a CPU version of the mesh.
        """
        def rotate_x(vertices):
            angle_rad = np.radians(90)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
            return vertices @ rotation_matrix.T
        self.vertex_channels = {k: srgb_to_linear(v) for k, v in self.vertex_channels.items()}
        return TriMesh(
        verts=self.verts.detach().cpu().numpy(),
            faces=self.faces.cpu().numpy(),
            vertex_channels=(
                {k: v.detach().cpu().numpy() for k, v in self.vertex_channels.items()}
                if self.vertex_channels is not None
                else None
            ),
            face_channels=(
                {k: v.detach().cpu().numpy() for k, v in self.face_channels.items()}
                if self.face_channels is not None
                else None
            ),
        )
