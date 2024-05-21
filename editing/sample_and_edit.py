import argparse
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np
import gc
import os
from shap_e.util.image_util import load_image
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images,decode_latent_mesh, gif_widget
from shap_e.diffusion.k_diffusion import GaussianToKarrasDenoiser

# Argument Parsing
parser = argparse.ArgumentParser(description='Run the model with specified configurations.')
parser.add_argument('--guide_image', type=str, required=True, help='Path to the guide image file.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model file.')
parser.add_argument('--inversion_method', type=str, choices=['ddim', 'ddpm'], required=True, help='Method of inversion to use (ddim or ddpm).')
parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output GIF.')
 
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device).eval()
model_path = args.model_path
guide_image = args.guide_image

batch = load_or_create_multimodal_batch(
    device,
    model_path=model_path,
    mv_light_mode="basic",
    mv_image_size=256,
    cache_dir=f"example_data/{os.path.basename(model_path)}",
    verbose=True,  # this will show Blender output during renders
)

with torch.no_grad():
    latent = xm.encoder.encode_to_bottleneck(batch)

from shap_e.diffusion.sample import sample_latents, sample_latents_noised
# from shap_e.diffusion.gaussian_diffusion import ddim_inversion
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config,GaussianDiffusion
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
    # def ddpm_inversion(self, latent,sigma_min,steps,rho,device):
sigma_min,sigma_max,steps = 1e-3, 160, 120
try:
    # raise Exception
    latents_noised = torch.load(f'latents_cached/{os.path.basename(model_path)}_{args.inversion_method}.pt')
except:
    if args.inversion_method == 'ddim':
        latents_noised = diffusion.ddim_inversion(model=model, latent=latent, clip_denoised=True, model_kwargs=dict(texts=["a flower pot"] * 1))
    else:
        latents_noised = GaussianToKarrasDenoiser(model,diffusion).ddpm_inversion( latent=latent,sigma_min=sigma_min,sigma_max=sigma_max,steps=steps,rho=7.0,device=device)
    
    if not os.path.exists('latents_cached'):
        os.mkdir('latents_cached')
    torch.save(latents_noised, f'latents_cached/{os.path.basename(model_path)}_{args.inversion_method}.pt')

guide_image_loaded_tensor = torch.from_numpy(np.array(Image.open(guide_image).convert('RGB'))).permute(2, 0, 1).to(device).unsqueeze(0).float()
guide_image_loaded_tensor = Resize(128)(guide_image_loaded_tensor).squeeze() / 255


# model = load_model('image300M', device=device)
image = load_image(args.guide_image)
# /home/yiftach.ede/shap-e/
batch_size = 1

latents_new = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=12,
    # model_kwargs=dict(images=[image] * batch_size),
    model_kwargs=dict(texts=["a flower pot with red flowers"] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min = sigma_min,
    sigma_max = 24 ,
    s_churn=0,
    noise=latents_noised[100].repeat(batch_size, 1),
)
with torch.no_grad():
    cameras = create_pan_cameras(512,device)
    for i, latent_code in enumerate(latents_new):
        # latent_code = latent_code.reshape(1024,1024) 
        # latent_code[768:] = 20*torch.log(latent_code[768:])
        # latent_code = latent_code.reshape(-1)
        decoder_output = decode_latent_images(xm, latent_code.float(), cameras, rendering_mode='stf')
        arr = decoder_output.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        images = [Image.fromarray(x) for x in arr]
        import io
        import base64
        import plotly.express as px

        gif_io = io.BytesIO()
        images[0].save(gif_io, format='GIF', save_all=True, append_images=images[1:], duration=200, loop=0)
        gif_io.seek(0)  # Go to the start of the GIF byte stream

        # Convert GIF to base64 for embedding
        gif_base64 = base64.b64encode(gif_io.read()).decode('utf-8')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Further down in your script, after generating the images list
        gif_path = os.path.join(args.output_dir, f'output{i}.gif')
        images[0].save(gif_path, format='GIF', save_all=True, append_images=images[1:], duration=200, loop=0)

        print(f"Saved GIF to {gif_path}")
        # Display the GIF with Plotly
        fig = px.imshow([[]], title='Animated GIF')
        fig.update_layout(images=[dict(source='data:image/gif;base64,' + gif_base64, xref="paper", yref="paper", x=0, y=1, sizex=1, sizey=1, sizing="contain", layer="above")])
        fig.update_xaxes(visible=False).update_yaxes(visible=False)

        fig.show()
        