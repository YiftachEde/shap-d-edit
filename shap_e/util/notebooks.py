import base64
import io
from typing import Union

import ipywidgets as widgets
import numpy as np
import torch
from PIL import Image

from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.rendering.torch_mesh import TorchMesh
from shap_e.util.collections import AttrDict


def create_pan_cameras(size: int, device: torch.device, n_samples=-1,n_angles=20) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=n_angles+1)[:-1]:
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    if n_samples > 0:
        
        indices = np.arange(0, n_samples)
        origins = [origins[i] for i in indices]
        xs = [xs[i] for i in indices]
        ys = [ys[i] for i in indices]
        zs = [zs[i] for i in indices]
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )

import numpy as np
import torch
import math

import math
import numpy as np
import torch

def create_custom_cameras(size: int, device: torch.device, azimuths: list, elevations: list, 
                          fov_degrees: float,distance) -> DifferentiableCameraBatch:
    # Object is in a 2x2x2 bounding box (-1 to 1 in each dimension)
    object_diagonal =  distance # Correct diagonal calculation for the cube
    
    # Calculate radius based on object size and FOV
    fov_radians = math.radians(fov_degrees)
    radius = (object_diagonal / 2) / math.tan(fov_radians / 2)  # Correct radius calculation

    origins = []
    xs = []
    ys = []
    zs = []
    
    for azimuth, elevation in zip(azimuths, elevations):
        azimuth_rad = np.radians(azimuth-90)
        elevation_rad = np.radians(elevation)
        
        # Calculate camera position
        x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = radius * np.sin(elevation_rad)
        origin = np.array([x, y, z])
        
        # Calculate camera orientation
        z_axis = -origin / np.linalg.norm(origin)  # Point towards center
        x_axis = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0])
        y_axis = np.cross(z_axis, x_axis)
        
        origins.append(origin)
        zs.append(z_axis)
        xs.append(x_axis)
        ys.append(y_axis)

    return DifferentiableCameraBatch(
        shape=(1, len(origins)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=fov_radians,
            y_fov=fov_radians,
        ),
    )


# @torch.no_grad()
def decode_latent_images(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
    cameras: DifferentiableCameraBatch,
    rendering_mode: str = "stf",
    background_color: torch.Tensor = torch.tensor([255.0, 255.0, 255.0], dtype=torch.float32),
):
    decoded = xm.renderer.render_views(
        AttrDict(cameras=cameras),
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode=rendering_mode, render_with_direction=False),
    )
    bg_color = background_color.to(decoded.channels.device)
    images = bg_color * decoded.transmittance + (1 - decoded.transmittance) * decoded.channels

    # arr = decoded.channels.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return images

# @torch.no_grad()
def decode_latent_mask_by_bbox(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
    cameras: DifferentiableCameraBatch,
    rendering_mode: str = "stf",
    bbox: torch.Tensor = torch.tensor([[-1.0, -1.0,-1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
):
    grads = xm.renderer.generate_latent_mask(
        encoder=xm.encoder,
        latents = latent[None],
        options = AttrDict(bbox=bbox, render_with_direction=False, rendering_mode=rendering_mode),
    )
    return grads[0]
    # bg_color = background_color.to(decoded.channels.device)
    # images = bg_color * decoded.transmittance + (1 - decoded.transmittance) * decoded.channels

    # arr = decoded.channels.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return decoded.channels



def decode_latent_mesh(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
) -> TorchMesh:
    decoded = xm.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, latent.device)),  # lowest resolution possible
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]


def gif_widget(images):
    writer = io.BytesIO()
    images[0].save(
        writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0
    )
    writer.seek(0)
    data = base64.b64encode(writer.read()).decode("ascii")
    return widgets.HTML(f'<img src="data:image/gif;base64,{data}" />')
