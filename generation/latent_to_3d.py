import math
import os
import subprocess
import torch
from pathlib import Path
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh
import argparse
from multiprocessing import Process, Queue, current_process, set_start_method
def scene_meshes():
    import bpy
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def run_command(command):
    subprocess.run(command)
    
def process_model(input_queue, output_dir):
    while True:
        path = input_queue.get()
        if path is None:
            break  # Exit signal

        file_name_no_ext = Path(path).stem
        output_folder = f"{output_dir}/{file_name_no_ext}"
        os.makedirs(output_folder, exist_ok=True)
        
        command = [
            '/home/yiftach/miniconda3/envs/instantmesh/bin/render_blender',
            '--object_file', path,
            '--output_dir', output_folder,
            '--azimuths', '30', '90', '150', '210', '270', '330',
            '--elevations', '20', '-10', '20', '-10', '20', '-10',
            '--width', '512', '--height', '512',
        ]
        
        run_command(command)
        os.remove(path)
        print(f"Processed {path} to {output_folder}")


def worker(input_queue, output_queue, device):
    xm = load_model('transmitter', device=device)  # Load model for each worker

    while True:
        latent_path = input_queue.get()
        if latent_path is None:
            output_queue.put(None)
            break  # Exit signal
        
        print(f"{current_process().name} processing {latent_path}")
        latents = torch.load(latent_path).to(device)
        for latent in latents:
            with torch.no_grad():
                mesh = decode_latent_mesh(xm, latent.float()).tri_mesh()
                uuid = os.path.basename(latent_path).split('.')[0]
                mesh_path = f"/tmp/{uuid}.obj"
                with open(mesh_path, 'w') as f:
                    mesh.write_obj(f)
                output_queue.put(mesh_path)
                os.remove(latent_path)
            print(f"Generated {mesh_path}")

def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    argparser = argparse.ArgumentParser()
    argparser.add_argument('output_path', type=str, default='output path here')
    args = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = args.output_path
    latents = set([latent[:-3] for latent in os.listdir("comp_code/Cap3D_latentcodes/")])
    # uuids_in_dataset = set(os.listdir("/mnt/net_store/refinement_dataset_subset_relighted"))
    uuids_done = set(os.listdir(output_dir))
    # print(latents)
    # print(uuids_done)
    # exit(0)
    # total_len = len(latents)
    # latents = latents - uuids_in_dataset
    # print(f"Processing {len(latents)} latents, removed {total_len - len(latents)} already in dataset")
    total_len = len(latents)
    latents = latents - uuids_done
    print(f"Processing {len(latents)} latents, removed {total_len - len(latents)} already done")
    latent_paths = [f"comp_code/Cap3D_latentcodes/{fname}.pt" for fname in latents]
    
    # Create queues
    task_queue = Queue()
    result_queue = Queue(400)
    # Enqueue all latent paths
    for path in latent_paths:
        task_queue.put(path)
    
    # Start worker processes
    num_workers = os.cpu_count()
    workers = [Process(target=worker, args=(task_queue, result_queue, device)) for _ in range(5)]
    model_workers = [Process(target=process_model, args=(result_queue, output_dir)) for _ in range(20)]

    for w in workers:
        w.start()
    for mw in model_workers:
        mw.start()

    # Signal the end of tasks
    for _ in range(num_workers):
        task_queue.put(None)

    # Main process converts OBJ to GLB using Blender
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # Join all workers
    for w in workers:
        w.join()
    for mw in model_workers:
            mw.join()
if __name__ == '__main__':
    main()