"""
Based on: https://github.com/crowsonkb/k-diffusion

Copyright (c) 2022 Katherine Crowson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import torch as th
from tqdm import tqdm
from .gaussian_diffusion import GaussianDiffusion, mean_flat

def _extract_into_tensor(arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + th.zeros(broadcast_shape, device=timesteps.device)

class KarrasDenoiser:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, sigmas, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        terms = {}

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        c_skip, c_out, _ = [append_dims(x, dims) for x in self.get_scalings(sigmas)]
        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)
        target = (x_start - c_skip * x_t) / c_out

        terms["mse"] = mean_flat((model_output - target) ** 2)
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised


class GaussianToKarrasDenoiser:
    def __init__(self, model, diffusion):
        from scipy import interpolate

        self.model = model
        self.diffusion = diffusion
        # print(diffusion.alphas_cumprod.shape)
        self.alpha_cumprod_to_t = interpolate.interp1d(
            diffusion.alphas_cumprod, np.arange(0, diffusion.num_timesteps)
        )

    @th.no_grad()
    def ddpm_inversion(self, latent,sigma_min,sigma_max,steps,device,rho=7.0):
        sigmas = reversed(get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device))[:,None]
        print(sigmas.shape)
        latent=     latent + th.randn(*latent.shape, device=device) * sigmas 
        return latent


    
    def sigma_to_t(self, sigma):
        alpha_cumprod = 1.0 / (sigma**2 + 1)
        if alpha_cumprod > self.diffusion.alphas_cumprod[0]:
            return 0
        elif alpha_cumprod <= self.diffusion.alphas_cumprod[-1]:
            return self.diffusion.num_timesteps - 1
        else:
            return int(self.alpha_cumprod_to_t(alpha_cumprod))
   
    

    def condition_score(self, cond_fn, p_mean_var, x, t, **model_kwargs):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(p_mean_var["pred_xstart"], **(model_kwargs or {}))

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def denoise(self, x_t, sigmas, clip_denoised=True, model_kwargs=None):
        t = th.tensor(
            [self.sigma_to_t(sigma) for sigma in sigmas.cpu().numpy()],
            dtype=th.long,
            device=sigmas.device,
        )
        c_in = append_dims(1.0 / (sigmas**2 + 1) ** 0.5, x_t.ndim)
        out = self.diffusion.p_mean_variance(
            self.model, x_t * c_in, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        return None, out["pred_xstart"]


def karras_sample(*args, **kwargs):
    last = None
    noise = kwargs.pop("noise", None)
    for x in karras_sample_progressive(*args,noise=noise, **kwargs):
        last = x["x"]
        noise = last
    return last

def karras_sample_progressive(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    guidance_scale=0.0,
    noise=None,
    guidance_fn = None
):
    # print(steps)
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
    # print(sigma_min,sigma_max,rho)
    x_T = th.randn(*shape, device=device) * sigma_max if noise is None else noise
    sample_fn = {"heun": sample_heun, "dpm": sample_dpm, "ancestral": sample_euler_ancestral}[
        sampler
    ]

    if sampler != "ancestral":
        sampler_args = dict(s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, guidance_fn=guidance_fn, diffusion=diffusion)
    else:
        sampler_args = {}

    if isinstance(diffusion, KarrasDenoiser):

        def denoiser(x_t, sigma):
            _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised

    elif isinstance(diffusion, GaussianDiffusion):
        model = GaussianToKarrasDenoiser(model, diffusion)
        def denoiser(x_t, sigma):
            _, denoised = model.denoise(
                x_t, sigma, clip_denoised=clip_denoised, model_kwargs=model_kwargs
            )
            return denoised

    else:
        raise NotImplementedError

    if guidance_scale != 0 and guidance_scale != 1:

        def guided_denoiser(x_t, sigma):
            x_t = th.cat([x_t, x_t], dim=0)
            sigma = th.cat([sigma, sigma], dim=0)
            x_0 = denoiser(x_t, sigma)
            cond_x_0, uncond_x_0 = th.split(x_0, len(x_0) // 2, dim=0)
            x_0_cfg = uncond_x_0 + guidance_scale * (cond_x_0 - uncond_x_0)
            p_mean_var = {'pred_xstart': cond_x_0, 'mean': cond_x_0}
            t = th.tensor(model.sigma_to_t(sigma[0:1].cpu())).to(x_t.device)
            cond_kwargs = model_kwargs.copy()
            cond_kwargs['cond_or_uncond'] = 'noisy'

            # x_0 = x_0.clamp(-1,1)
            return x_0_cfg

    else:
        guided_denoiser = denoiser

    for obj in sample_fn(
        guided_denoiser,
        x_T,
        sigmas,
        progress=progress,
        **sampler_args,
    ):
        if isinstance(diffusion, GaussianDiffusion):
            yield diffusion.unscale_out_dict(obj)
        else:
            yield obj


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, progress=False):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "pred_xstart": denoised}
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + th.randn_like(x) * sigma_up
    yield {"x": x, "pred_xstart": x}


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    guidance_fn = None,
    diffusion = None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        )
        eps = th.randn_like(x) * s_noise
        # guidance_fn = guidance_fn if guidance_fn is not None else lambda _ : 0
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "pred_xstart": denoised}
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d*dt
        else:
            # Heun's method
            # d = (
            x_2 = x + d*dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    yield {"x": x, "pred_xstart": x}


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        )
        eps = th.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised}
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    yield {"x": x, "pred_xstart": denoised}


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return th.cat([x, x.new_zeros([1])])
