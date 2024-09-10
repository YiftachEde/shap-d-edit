import argparse
import json
import torch
import os
from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from tqdm import tqdm
import numpy as np
import trimesh
import struct
from pygltflib import *

# Argument Parsing
parser = argparse.ArgumentParser(description='Run the model with specified configurations.')
parser.add_argument('models_json', type=str, help='Path to the model file.')
parser.add_argument('--output_dir', type=str, default='./edited_outputs', help='Directory to save output objects.')
parser.add_argument('--edit_strength', type=float, default=1.0, help='Strength of edit')
parser.add_argument('--num_edit_steps', type=int, default=128, help='Number of edit steps')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device).eval()
text_guidance_model = load_model('text300M', device=device)
edited_captions = json.load(open(args.models_json))
diffusion = diffusion_from_config(load_config('diffusion'))


sRGB_TO_LINEAR_RGB = True

def rotate_x(vertices, angle_deg):
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return vertices @ rotation_matrix.T


def obj_to_glb(obj_path,glb_path):

    mesh = trimesh.load_mesh(obj_path)

    # Rotate mesh vertices
    # mesh.vertices = rotate_x(mesh.vertices, -90)  # Example rotation
    # mesh.vertex_normals = rotate_x(mesh.vertex_normals, -90)

    # instantiate GLTF2
    gltf = GLTF2()
    gltf.asset = Asset(generator="Custom OBJ to GLTF Converter", version="2.0")
    gltf.scenes = [Scene()]
    gltf.nodes = [Node()]       # Mesh node
    gltf.meshes = [Mesh()]
    gltf.accessors = [Accessor() for _ in range(4)]     # faces, vertices, v_colors, v_norms
    gltf.materials = [Material()]
    gltf.bufferViews = [BufferView() for _ in range(4)]
    gltf.buffers = [Buffer()]

    # asset
    gltf.asset = Asset()
    gltf.asset.generator = "OBJwVC_to_glTF"
    gltf.asset.copyright = "Tomoaki Osada (ynyBonfennil)"

    # scene
    gltf.scene = 0

    # materials
    gltf.materials[0].pbrMetallicRoughness = PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.5)
    gltf.materials[0].emissiveFactor = [0.0, 0.0, 0.0]
    gltf.materials[0].name = "material 001-effect"

    # store faces
    # face (indices) data must be stored as a sequence of SCALAR value
    # hense accessor.type is set to SCALAR, and accessors.count is
    # 3 times the length of face vectors
    indices_chunk = b""
    for f in mesh.faces:
        indices_chunk += struct.pack("<III", *f)
    gltf.bufferViews[0].buffer = 0
    gltf.bufferViews[0].byteOffset = 0
    gltf.bufferViews[0].byteLength = len(indices_chunk)
    gltf.bufferViews[0].target = ELEMENT_ARRAY_BUFFER
    gltf.accessors[0].bufferView = 0
    gltf.accessors[0].byteOffset = 0
    gltf.accessors[0].componentType = UNSIGNED_INT
    gltf.accessors[0].normalized = False
    gltf.accessors[0].count = len(mesh.faces) * 3
    gltf.accessors[0].type = "SCALAR"

    # store vertices
    vertices_chunk = b""
    mesh.vertices = rotate_x(mesh.vertices, -90)
    mesh.vertex_normals = rotate_x(mesh.vertex_normals, -90)
    for v in mesh.vertices:
        vertices_chunk += struct.pack("<fff", *v)
    gltf.bufferViews[1].buffer = 0
    gltf.bufferViews[1].byteOffset = gltf.bufferViews[0].byteLength
    gltf.bufferViews[1].byteLength = len(vertices_chunk)
    gltf.bufferViews[1].target = ARRAY_BUFFER
    gltf.accessors[1].bufferView = 1
    gltf.accessors[1].byteOffset = 0
    gltf.accessors[1].componentType = FLOAT
    gltf.accessors[1].normalized = False
    gltf.accessors[1].count = len(mesh.vertices)
    gltf.accessors[1].type = "VEC3"
    gltf.accessors[1].max = list(np.max(mesh.vertices.T, axis=1))       # get the max value for each xyz
    gltf.accessors[1].min = list(np.min(mesh.vertices.T, axis=1))

    # store vertex colors
    vcolor_chunk = b""
    if sRGB_TO_LINEAR_RGB:          # sRGB to Linear RGB if needed.
        vc = mesh.visual.vertex_colors[:, :3] / 255
        vc = np.clip(vc, 0, None)
        vc[vc < 0.04045] = vc[vc < 0.04045] * (1.0 / 12.92)
        vc[vc >= 0.04045] = pow((vc[vc >= 0.04045] + 0.055) * (1.0 / 1.055), 2.4)
        mesh.visual.vertex_colors[:, :3] = vc * 255
    for vc in mesh.visual.vertex_colors:
        vc_rgb = vc[:3] / 255
        vcolor_chunk += struct.pack("<fff", *vc_rgb)
    gltf.bufferViews[2].buffer = 0
    gltf.bufferViews[2].byteOffset = gltf.bufferViews[1].byteOffset + gltf.bufferViews[1].byteLength
    gltf.bufferViews[2].byteLength = len(vcolor_chunk)
    gltf.bufferViews[2].target = ARRAY_BUFFER
    gltf.accessors[2].bufferView = 2
    gltf.accessors[2].byteOffset = 0
    gltf.accessors[2].componentType = FLOAT
    gltf.accessors[2].normalized = False
    gltf.accessors[2].count = len(mesh.visual.vertex_colors)
    gltf.accessors[2].type = "VEC3"

    vnorm_chunk = b""
    for vn in mesh.vertex_normals:
        vnorm_chunk += struct.pack("<fff", *vn)
    gltf.bufferViews[3].buffer = 0
    gltf.bufferViews[3].byteOffset = gltf.bufferViews[2].byteOffset + gltf.bufferViews[2].byteLength
    gltf.bufferViews[3].byteLength = len(vnorm_chunk)
    gltf.bufferViews[3].target = ARRAY_BUFFER
    gltf.accessors[3].bufferView = 3
    gltf.accessors[3].byteOffset = 0
    gltf.accessors[3].componentType = FLOAT
    gltf.accessors[3].normalized = False
    gltf.accessors[3].count = len(mesh.vertex_normals)
    gltf.accessors[3].type = "VEC3"

    # store buffer data
    gltf.identify_uri = BufferFormat.BINARYBLOB
    gltf._glb_data = indices_chunk + vertices_chunk + vcolor_chunk + vnorm_chunk
    gltf.buffers[0].byteLength = gltf.bufferViews[3].byteOffset + gltf.bufferViews[3].byteLength
    
    # mesh
    gltf.meshes[0].primitives = [
        Primitive(
            attributes=Attributes(
                POSITION=1,
                NORMAL=3,
                COLOR_0=2,
            ),
            indices=0,
            material=0
        )
    ]
    gltf.meshes[0].name = "Mesh"

    # assemble nodes
    gltf.nodes[0].mesh = 0
    gltf.nodes[0].name = "Mesh"

    gltf.scenes[0].nodes = [0]

    # export
    gltf.save_binary(glb_path)

def export_latent_mesh_to_obj(args, xm, uuid, i, latent_code):
    with torch.no_grad():
            # if args.output_type == "images":
        mesh = decode_latent_mesh(xm, latent_code.float()).tri_mesh()
        os.makedirs(f'{args.output_dir}/{uuid}', exist_ok=True)
        obj_path = f'{args.output_dir}/{uuid}/{i}.obj'
        glb_path = f'{args.output_dir}/{uuid}/{i}.glb'
        with open(f'{args.output_dir}/{uuid}/{i}.obj', 'w') as f:
            mesh.write_obj(f)
        obj_to_glb(obj_path, glb_path)

for uuid,model in tqdm(edited_captions.items()):
    try:
        metadata_path = f'{args.output_dir}/{uuid}/model.json'
        if os.path.exists(metadata_path):
            old_metadata = json.load(open(metadata_path))
        else:
            old_metadata = None
        model_path = model['path']
        print("cache path is",f"cached_data/{os.path.basename(model_path)}")
        batch = load_or_create_multimodal_batch(
            device,
            model_path=model_path,
            mv_light_mode="basic",
            mv_image_size=256,
            cache_dir=f"cached_data/{os.path.basename(model_path)}",
            verbose=True,  # this will show Blender output during renders
        )

        with torch.no_grad():
            latent = xm.encoder.encode_to_bottleneck(batch)


        zs_batch,latents_noised_batch = [],[]
        latents_new = []
        export_latent_mesh_to_obj(args, xm, uuid, -1, latent)
        for i,[caption,edit_strength] in enumerate(model['editing captions']):
            if old_metadata is not None and len(old_metadata['editing captions'])==len(model['editing captions']) and old_metadata['editing captions'][i][0] == caption and old_metadata['editing captions'][i][1] == edit_strength:
                print(f"Skipping {caption} with {edit_strength}")
                continue
            start_step = int(args.num_edit_steps*edit_strength)
            print(f"\nEditing {model['original caption']} with {caption}, edit strength {edit_strength}, starting at step {start_step}")
            _,zs,latents_noised = diffusion.ddpm_inversion(model=text_guidance_model,x0=latent,cfg_scale=12, clip_denoised=False, model_kwargs=dict(texts=[model['original caption']] * 1),num_inference_steps=args.num_edit_steps)
            latents_new_cur,_ = diffusion.inversion_reverse_process(text_guidance_model,latents_noised[start_step],prompts=[caption],cfg_scales=12,prog_bar=True,zs=zs[:start_step],clip_denoised=False)
            export_latent_mesh_to_obj(args, xm, uuid, i, latents_new_cur)
            
        

        json.dump(model, open(metadata_path, 'w'))
    except Exception as e:
        print(f"Error in {uuid}: {e}")
        continue