import json
import os
import re
import subprocess
from pathlib import Path 
import argparse
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np
from tqdm import tqdm

def run_command(command):
    subprocess.run(command)
    
def process_model(path, output_dir):
    parent_folder = Path(path).parent
    file_name_no_ext = Path(path).stem
    output_folder = f"{output_dir}/{file_name_no_ext}"
    os.makedirs(output_folder, exist_ok=True)
    
    command = [
        '/home/yiftach/miniconda3/envs/instantmesh/bin/render_blender',
        '--object_file', path, 
        '--output_dir', output_folder,
        '--azimuths', '0', '30', '90', '150', '210', '270', '330',
        '--elevations', '20', '20', '-10', '20', '-10', '20', '-10',
        '--width', '512', '--height', '512', 
    ]
        # print(command)
        
    run_command(command)
    return output_folder
    
def main():
    parser = argparse.ArgumentParser(description="Process a 3D model.")
    parser.add_argument("input_json", type=str, help="The path to the json file.")
    args = parser.parse_args()
    files = json.load(open(args.input_json))
    parent_dir = os.path.dirname(args.input_json)
    for entry in files['entries']:
        name =  entry['name']
        pattern = re.compile(rf"{name}_[0-9].obj")
        num_glbs_under_name_in_dir = len([f for f in os.listdir(parent_dir) if pattern.match(f) and f.endswith('.obj')])
        print(num_glbs_under_name_in_dir)
        obj_files = [f"{parent_dir}/{name}_{i}.obj" for i in range(num_glbs_under_name_in_dir)]
        print(name, f"has {num_glbs_under_name_in_dir} obj files")
        for obj in obj_files:
            process_model(obj, parent_dir)
        # output_folder = process_model(file['path'], files['output_dir'])
        # image_sequence = []
        
        # for i in range(120):
        #     image_path = f"{output_folder}/render_{i:04d}.png"
        #     image = Image.open(image_path)
            
        #     # Create a white background
        #     white_bg = Image.new("RGB", image.size, (255, 255, 255))
            
        #     # Composite the RGBA image onto the white background
        #     image = Image.alpha_composite(white_bg.convert("RGBA"), image).convert("RGB")
            
        #     image_sequence.append(np.array(image))
        #     os.remove(image_path)
        
        # # Create a video clip from the image sequence
        # clip = ImageSequenceClip(image_sequence, fps=30)

        # # Save the clip as an MP4 with compression
        # clip.write_videofile(f'{output_folder}/gt_render.mp4', codec="libx264", fps=30, bitrate="2000k")
        # print(f"Processed {file['path']}, wrote mp4 to path {output_folder}/gt_render.mp4")
    
if __name__ == "__main__":
    main()