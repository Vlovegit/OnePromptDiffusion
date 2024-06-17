from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from utils import SimpleDaamPipeline
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from utils import ptp_utils
import shutil
from torch.optim.adam import Adam
from PIL import Image
from utils import load_coco_dataset
import utils.supersecrets as ss
import neptune
import os
from neptune.types import File
import h5py as h5
from utils import detect_object
import pandas as pd
from datetime import datetime
import utils.config as env

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
experimentName = "Object Filter COCO Dataset"

if experimentName is not None:
    run = neptune.init_run(
        project=ss.neptune_project,
        api_token=ss.api_token,
        name=experimentName,
        tags=['Object Filter']
    )

#Process set of prompts at a time to avoid memory issues
NUM_PROMPTS = 2000

# Configuration
rec = 0

# Create the results directory
os.makedirs(f"{env.object_cocoids}/results/",exist_ok=True)
os.makedirs(f"{env.object_cocoids}/results_working/",exist_ok=True)
os.makedirs(f"{env.object_cocoids}/results_not_working/",exist_ok=True)
os.makedirs(f"{env.object_cocoids}/results_useless/",exist_ok=True)

for i in range(0,32):
    print(f'Processing set of prompts: {i*NUM_PROMPTS} to {(i+1)*NUM_PROMPTS}')
    coco_ids, prompts, negative_promts, images = load_coco_dataset(NUM_PROMPTS,i*NUM_PROMPTS)
    working_coco_ids = {}
    not_working_coco_ids = {}
    useless_coco_ids = {}
    discarded_coco_ids = {}
    for coco_id,prompt,negative_prompt in zip(coco_ids,prompts,negative_promts):
        print(f'Processing record no: {rec} COCO id: {coco_id} with prompt: {prompt} and negative prompt: {negative_prompt}')
        rec += 1
        if negative_prompt in prompt:
            if str(coco_id) in os.listdir(f"{env.object_cocoids}/results_working") or str(coco_id) in os.listdir(f"{env.object_cocoids}/results_not_working") or str(coco_id) in os.listdir(f"{env.object_cocoids}/results_useless"):
                print(f'COCO id: {coco_id} already processed')
                continue
            image = ldm_stable(prompt=prompt).images[0]       
            os.makedirs(f"{env.object_cocoids}/results/{coco_id}",exist_ok=True)
            image_path = f"{env.object_cocoids}/results/{coco_id}/{negative_prompt}.png"
            image.save(image_path)
            prompt_list = prompt.split(' ')
            object_present, mask_area_p, area_threshold_p = detect_object(image_path, prompt_list, negative_prompt, prompt, True)
            if object_present:
                image_ng = ldm_stable(prompt=prompt,negative_prompt=negative_prompt).images[0]
                image_ng_path = f"{env.object_cocoids}/results/{coco_id}/{negative_prompt}_ng.png"
                image_ng.save(image_ng_path)
                object_present, mask_area_ng, area_threshold_ng = detect_object(image_ng_path, prompt_list, negative_prompt, prompt, False)
                if not object_present:
                    os.makedirs(f"{env.object_cocoids}/results_working/{coco_id}",exist_ok=True)
                    image.save(f"{env.object_cocoids}/results_working/{coco_id}/{negative_prompt}.png")
                    image_ng.save(f"{env.object_cocoids}/results_working/{coco_id}/{negative_prompt}_ng.png")
                    run[f'output/images/{coco_id}'].append(File.as_image(image),description=f'Image Positive Prompt Only : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id} \n Area: {mask_area_p} \n Threshold: {area_threshold_p}')
                    run[f'output/images/{coco_id}'].append(File.as_image(image_ng),description=f'Image with Negative Prompt : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id} \n Area: {mask_area_ng} \n Threshold: {area_threshold_ng}')
                    print('Object present in positive prompt only image and absent in with negative prompt image')
                    #Save the coco_id with threshold prompt and negative prompt info to the csv
                    working_coco_ids[coco_id] = (prompt,negative_prompt,mask_area_p,area_threshold_p,mask_area_ng,area_threshold_ng)
                else:
                    run[f'output/images_not_working/{coco_id}'].append(File.as_image(image),description=f'Image Positive Prompt Only : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id} \n Area: {mask_area_p} \n Threshold: {area_threshold_p}')
                    run[f'output/images_not_working/{coco_id}'].append(File.as_image(image_ng),description=f'Image with Negative Prompt : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id} \n Area: {mask_area_ng} \n Threshold: {area_threshold_ng}')
                    os.makedirs(f"{env.object_cocoids}/results_not_working/{coco_id}",exist_ok=True)
                    image.save(f"{env.object_cocoids}/results_not_working/{coco_id}/{negative_prompt}.png")
                    image_ng.save(f"{env.object_cocoids}/results_not_working/{coco_id}/{negative_prompt}_ng.png")
                    not_working_coco_ids[coco_id] = (prompt,negative_prompt,mask_area_p,area_threshold_p,mask_area_ng,area_threshold_ng)
                    print('Object present in both images')
            else:
                print('Object not present in postive only prompt image')
                run[f'output/images_useless/{coco_id}'].append(File.as_image(image),description=f'Image Positive Prompt Only : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id} \n Area: {mask_area_p} \n Threshold: {area_threshold_p}')
                os.makedirs(f"{env.object_cocoids}/results_useless/{coco_id}",exist_ok=True)
                image.save(f"{env.object_cocoids}/results_useless/{coco_id}/{negative_prompt}.png")
                useless_coco_ids[coco_id] = (prompt,negative_prompt,mask_area_p,area_threshold_p)
        else:
            print('Negative prompt not present in positive prompt')
            discarded_coco_ids[coco_id] = (prompt,negative_prompt)

    print(f'Working COCO ids: {working_coco_ids}')
    print(f'Not Working COCO ids: {not_working_coco_ids}')
    print(f'Useless COCO ids: {useless_coco_ids}')

    working_df = pd.DataFrame(working_coco_ids)
    not_working_df = pd.DataFrame(not_working_coco_ids)
    useless_df = pd.DataFrame(useless_coco_ids)
    discarded_df = pd.DataFrame(discarded_coco_ids)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'{env.object_cocoids}/results_{timestamp}.xlsx'

    with pd.ExcelWriter(filename) as writer:
        working_df.to_excel(writer, sheet_name='working')
        not_working_df.to_excel(writer, sheet_name='not_working')
        useless_df.to_excel(writer, sheet_name='useless')
        discarded_df.to_excel(writer, sheet_name='discarded')

    print('Save to the file')

    