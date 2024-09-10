import math
import bpy
import torch
import json
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import argparse

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('generate_captions_json', type=str, default='caption file path')
    argparser.add_argument('--guidance_scale', type=float, default=15)
    argparser.add_argument('--batch_size', type=int, default=4)
    args = argparser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.load_state_dict(torch.load('shapE_finetuned_with_825kdata.pth', map_location=device)['model_state_dict'])
    diffusion = diffusion_from_config(load_config('diffusion'))
    captions = json.load(open(args.generate_captions_json))
    output_dir = captions['output_dir']
    entries = captions['entries']
    batch_size = args.batch_size
    guidance_scale = args.guidance_scale
    for entry in entries:    
        prompt = entry['caption']
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
            karras_steps=128,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        import os
        for i, latent in enumerate(latents[0]):
            os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

            with torch.no_grad():
                mesh = decode_latent_mesh(xm, latent.float()).tri_mesh()
                mesh_path = f"{output_dir}/{entry['name']}_{i}.obj"
                with open(mesh_path, 'w') as f:
                    mesh.write_obj(f)
                bpy.ops.wm.read_factory_settings(use_empty=True)

                # Import the OBJ file
                bpy.ops.wm.obj_import(filepath=mesh_path)
                print("Imported OBJ file successfully")
                glb_path = f"{output_dir}/{entry['name']}_{i}.glb"
                for obj in scene_meshes():
    
                    obj.rotation_euler[0] += math.radians(270)
                    obj.rotation_euler[2] += math.radians(0)
                # Export the scene as a GLB file, disable Draco compression
                bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB')
                print(f"Saved {glb_path}")
                    # exit(0)
                    
if __name__ == '__main__':
    main()