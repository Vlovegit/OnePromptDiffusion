from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
import sys
 
sys.path.insert(1, '/home/myid/vg80700/gits/OnePromptDiffusion/utils')

# from ..utils import SimpleDaamPipeline
from load_coco_set import load_coco_dataset
from pipes import SimpleDaamPipeline
# from utils import SimpleDaamPipeline
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from ptp_utils import *
import shutil
from torch.optim.adam import Adam
from PIL import Image
import supersecrets as ss
import neptune
import os
from neptune.types import File
import h5py as h5
from clip_seg import detect_object
import pandas as pd
from datetime import datetime
import config as env

# Load the model

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ss.huggingface_token
LOW_RESOURCE = False 
NUM_DDIM_STEPS = env.NUM_DDIM_STEPS
GUIDANCE_SCALE = env.GUIDANCE_SCALE
MAX_NUM_WORDS = 77
device = torch.device(env.device) if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = SimpleDaamPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler, seed=2551055002497238).to(device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

#Neptune Configuration
experimentName = "Quality Filter COCO Dataset"

if experimentName is not None:
    run = neptune.init_run(
        project=ss.neptune_project,
        api_token=ss.neptune_api_token,
        name=experimentName,
        tags=['Quality Filter']
    )

#Process set of prompts at a time to avoid memory issues
NUM_PROMPTS = 50

# Configuration
i = 0

negative_prompts_list = [
    "low resolution",
    "blurry",
    "overexposed",
    "noise",
    "grainy",
    "unnatural",
    "distorted",
    "floating",
    "inappropriate",
    "clipping",
    "ugly"
]

prompt_list = []

for i in range(0,1):
    print(f'Processing set of prompts: {i*NUM_PROMPTS} to {(i+1)*NUM_PROMPTS}')
    coco_ids, prompts, negative_promts, images = load_coco_dataset(NUM_PROMPTS,i*NUM_PROMPTS)
    working_coco_ids = {}
    for coco_id,prompt,negative_prompt in zip(coco_ids,prompts,negative_promts):
        #create a copies of the prompt based on length of negative prompts list   
        os.makedirs(f"{env.quality_cocoids}/{coco_id}",exist_ok=True)
        image = ldm_stable(prompt=prompt).images[0]
        image_path = f"{env.quality_cocoids}/{coco_id}/pos.png"
        image.save(image_path)
        run[f'output/images/{coco_id}'].append(File.as_image(image),description=f'Image Positive Prompt Only : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
        # randomly select a negative prompt from the list
        negative_prompt = str(np.random.choice(negative_prompts_list))
        print(f'Processing COCO id: {coco_id} with prompt: {prompt} and negative prompt: {negative_prompt}')
        image_quality = ldm_stable(prompt=prompt,negative_prompt=negative_prompt).images[0]       
        image_ng_path = f"{env.quality_cocoids}/{coco_id}/{negative_prompt}.png"
        image_quality.save(image_ng_path)
        run[f'output/images/{coco_id}'].append(File.as_image(image_quality),description=f'Image with Negative Prompt : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
        #Save the coco_id with threshold prompt and negative prompt info to the csv
        working_coco_ids[coco_id] = {'prompt':prompt,'negative_prompt':negative_prompt}

print(f'Working COCO ids: {working_coco_ids}')

working_df = pd.DataFrame(working_coco_ids)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f'{env.quality_cocoids}/results_quality_{timestamp}.xlsx'

with pd.ExcelWriter(filename) as writer:
    working_df.to_excel(writer, sheet_name='working')

print('Save to the file')

    