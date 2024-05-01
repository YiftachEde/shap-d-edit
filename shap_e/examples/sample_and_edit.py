import torch
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np
import gc
from shap_e.util.image_util import load_image
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images,decode_latent_mesh, gif_widget
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device).eval()
# for param in xm.parameters():
    # param.requires_grad = False
model_path = "/home/yiftach/main/Research/spic-e/shap-e-data/ModelNet40_ply/chair/train/chair_0142.ply"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model = clip_model.eval().to(device)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
guide_image = "/home/yiftach/main/Research/shap-e/guide_image.png"
# This may take a few minutes, since it requires rendering the model twice
# in two different modes.
batch = load_or_create_multimodal_batch(
    device,
    model_path=model_path,
    mv_light_mode="basic",
    mv_image_size=256,
    cache_dir="example_data/chair",
    verbose=True, # this will show Blender output during renders
)
with torch.no_grad():
    latent = xm.encoder.encode_to_bottleneck(batch)

    # render_mode = 'nerf' # you can change this to 'nerf'
    # size = 512 # recommended that you lower resolution when using nerf

    # cameras = create_pan_cameras(size, device)
    # images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    # # display(gif_widget(images))
from shap_e.diffusion.sample import sample_latents, sample_latents_noised
# from shap_e.diffusion.gaussian_diffusion import ddim_inversion
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config,GaussianDiffusion
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
try:
    latents_noised = torch.load('latents_cached/chair_0142.pt')
except:    
    latents_noised = diffusion.ddim_inversion(model=model,latent=latent,clip_denoised=True, model_kwargs=dict(texts=["a table"] * 1))
    if os.path.exists('latents_cached') == False:
        os.mkdir('latents_cached')
    torch.save(latents_noised, 'latents_cached/chair_0142.pt')
diffusion = diffusion_from_config(load_config('diffusion'))

batch_size = 1
guidance_scale = 25.0
diffusion.num_timesteps = 20
diffusion.alphas_cumprod = diffusion.alphas_cumprod[:20]
n = 0
guide_image_loaded_tensor = torch.from_numpy(np.array(Image.open(guide_image).convert('RGB'))).permute(2, 0, 1).to(device).unsqueeze(0).float()
guide_image_loaded_tensor = Resize(128)(guide_image_loaded_tensor).squeeze()/255
def guide_model_via_loss(denoised,embedding=None,embeddings=None,cond_or_uncond="cond"):
    with torch.enable_grad():
        global n
        x_in = denoised[0:1].detach().requires_grad_(True).squeeze()
        cameras = create_pan_cameras(128,device,1)
        # cameras_sampled = cameras[:,camear_sampled_index:camear_sampled_index+1]
        decoded_output = decode_latent_images(xm, x_in,cameras, rendering_mode='nerf')
        image = decoded_output.permute(0,1,4,2,3)/255
        loss = torch.nn.functional.mse_loss(image.squeeze(),guide_image_loaded_tensor.squeeze())
        grad = torch.autograd.grad(loss, x_in)[0]
        # torch.cuda.empty_cache()
        # gc.collect()
        return 1*grad*1e6
    # # with open(f'example_mesh.ply', 'wb') as f:
    #     # mesh.tri_mesh().write_ply(f)

    return 0
# model = load_model('image300M', device=device)
image = load_image("guide_image2.png")

latents_new = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=12.5,
    # model_kwargs=dict(images=[image] * batch_size),
    model_kwargs=dict(texts=["a leather chair"] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
    noise=latents_noised[20].expand(batch_size, -1),
    guidance_fn = guide_model_via_loss
)
with torch.no_grad():
    cameras = create_pan_cameras(512,device)
    for i, latent_code in enumerate(latents_new):
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

        # Display the GIF with Plotly
        fig = px.imshow([[]], title='Animated GIF')
        fig.update_layout(images=[dict(source='data:image/gif;base64,' + gif_base64, xref="paper", yref="paper", x=0, y=1, sizex=1, sizey=1, sizing="contain", layer="above")])
        fig.update_xaxes(visible=False).update_yaxes(visible=False)

        fig.show()
