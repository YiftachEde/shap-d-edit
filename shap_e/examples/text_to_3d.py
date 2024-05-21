import torch
from PIL import Image

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
batch_size = 16
guidance_scale = 12.5
prompt = "shark"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=120,
    s_churn=3,
)
render_mode = 'stf' # you can change this to 'stf'
size = 256 # this is the size of the renders; higher values take longer to render.
import os
cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    dir_path = f'output_images/cloth_{i}'
    os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist

    with torch.no_grad():
        decoder_output = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        arr = decoder_output.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        images = [Image.fromarray(x) for x in arr]
        for j, image in enumerate(images):
            image_path = os.path.join(dir_path, f'shark_{i}_{j}.png')
            image.save(image_path)
