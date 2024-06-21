from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
import sys
 
sys.path.insert(1, '/home/myid/vg80700/gits/OnePromptDiffusion/utils')
sys.path.insert(2, '/home/myid/vg80700/gits/OnePromptDiffusion/mergeTextOptimization')

# from ..utils import SimpleDaamPipeline
from load_coco_set import load_coco_ids
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
from modified_nto import MergeTextOptimization as mto


scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ss.huggingface_token
LOW_RESOURCE = False 
NUM_DDIM_STEPS = env.NUM_DDIM_STEPS
GUIDANCE_SCALE = env.GUIDANCE_SCALE
MAX_NUM_WORDS = 77
device = torch.device(env.device) if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = SimpleDaamPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler, seed=2551055002497238).to(device)
mto_inversion = mto(ldm_stable, env.NUM_DDIM_STEPS, env.GUIDANCE_SCALE, env.device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

#Neptune Configuration
experimentName = "Quality Dataset Creation"

if experimentName is not None:
    run = neptune.init_run(
        project=ss.neptune_project,
        api_token=ss.neptune_api_token,
        name=experimentName,
        tags=['quality_dataset']
    )

BATCH_SIZE = env.file_batchsize

coco_id_list = os.listdir(env.quality_cocoids)

# coco_id_list = coco_id_list[:10]

coco_ids, prompts, negative_promts, images = load_coco_ids(coco_id_list)

data = {
    "coco_id": coco_ids,
    "prompt": prompts,
    "negative_prompt": negative_promts,
    "image": images
}

df = pd.DataFrame(data)

os.makedirs(env.quality_dataset, exist_ok=True)
DATA_FILE = f"{env.quality_dataset}/dataset_quality_0.h5"
print(f'Creating file {DATA_FILE}')
str_dtype = h5.string_dtype(encoding='utf-8', length=10)
str_prompt_dtype = h5.string_dtype(encoding='utf-8', length=300)
str_np_dtype = h5.string_dtype(encoding='utf-8', length=50)
img_dtype = h5.special_dtype(vlen=np.dtype('uint8'))
i = 0
count = 0
for coco_id in coco_id_list:
    print(f'Processing record no {count} out of {len(coco_id_list)}')
    count = count + 1
    if int(coco_id) in df['coco_id'].values:
        specific_data = df[df['coco_id'] == int(coco_id)]
        prompt = specific_data['prompt'].values[0]
        negative_prompt = specific_data['negative_prompt'].values[0]
        image_pos_path = f"{env.quality_cocoids}/{coco_id}/pos.png"
        image_pos = Image.open(image_pos_path)
        coco_file_list = os.listdir(f'/home/myid/vg80700/gits/results_quality/{coco_id}')
        neg_prompt_files = [item for item in coco_file_list if item != 'pos.png']
        image_neg_path = f"/home/myid/vg80700/gits/results_quality/{coco_id}/{neg_prompt_files[0]}"
        image_neg = Image.open(image_neg_path)
        print(f'Processing {coco_id}')
        (image_gt, image_enc), x_t, merged_embeddings, cond_embeddings, uncond_embeddings, latents = mto_inversion.invert(image_neg_path, prompt, negative_prompt, offsets=(0,0,200,0), verbose=True)
        image_inv, x_t1 = mto_inversion.run_and_display_merged(merged_embeddings[-1], ldm_stable, run_baseline=False, latent=x_t, verbose=False)
        lpips_score = compute_lpips(image_gt, image_inv[0])
        image_inv = Image.fromarray(image_inv[0].astype('uint8'), 'RGB')
        image_inv.save(f"{env.quality_cocoids}/{coco_id}/{negative_prompt}_inv.png")
        
        run[f'output/images/{coco_id}'].append(image_pos,description=f'Image Positive Prompt Only : for \n prompt:{prompt}\n COCO id: {coco_id}')
        run[f'output/images/{coco_id}'].append(image_neg,description=f'Image with Negative Prompt : for \n prompt:{prompt}\n negative prompt: {negative_prompt}\n COCO id: {coco_id}')
        run[f'output/images/{coco_id}'].append(File.as_image(image_inv),description=f'Image Inversion with Negative Prompt : for \n prompt:{prompt}\n negative prompt: {negative_prompt}\n COCO id: {coco_id} \n LPIPS Score: {lpips_score}')

        if lpips_score < 0.25:
            print(f'LPIPS Score for {coco_id} is {lpips_score} and is less than 0.25, saving in dataset')
            # Save to the dataset
            if i==0 or i%BATCH_SIZE==0:
                file = h5.File(DATA_FILE, 'w')
            else:
                file = h5.File(DATA_FILE, 'a')

            for j in range(len(merged_embeddings)):
                merged_embeddings[j] = merged_embeddings[j].detach().cpu().numpy()
            for k in range(len(latents)):
                latents[k] = latents[k].detach().cpu().numpy()

            cond_embeddings = cond_embeddings.detach().cpu().numpy()
            uncond_embeddings = uncond_embeddings.detach().cpu().numpy()
            image_neg_np = np.array(image_neg)
            image_pos_np = np.array(image_pos)
            image_inv_np = np.array(image_inv)

            if i==0:
                merged_dtype_latent = np.dtype([('idx', str_dtype),
                                        ('latents', float, latents[0].shape),
                                        ('merged_embeddings',float, merged_embeddings[-1].shape)])
                
                merged_dtype_all = np.dtype([('prompt', str_prompt_dtype),
                                        ('negative_prompt', str_np_dtype),
                                        ('cond_embeddings',np.float32, cond_embeddings.shape),
                                        ('uncond_embeddings',np.float32, uncond_embeddings.shape),
                                        ('merged_embeddings',np.float32, merged_embeddings[-1].shape),
                                        ('latents', np.float32, latents[0].shape),
                                        ('image_pos', image_pos_np.dtype, image_pos_np.shape),
                                        ('image_neg', image_neg_np.dtype, image_neg_np.shape),
                                        ('image_inv', image_inv_np.dtype, image_inv_np.shape),
                                        ('LPIPS', np.float32)])

            dset = file.create_dataset(f'{str(coco_id)}_lt', shape=(len(latents)-1), dtype=merged_dtype_latent, compression="gzip")
            for l in range(len(latents)-1):
                dset[l] = (str(l), latents[l],merged_embeddings[l])
                
            dset2 = file.create_dataset(str(coco_id), shape=1, dtype=merged_dtype_all, compression="gzip")
            dset2[0] = (prompt, negative_prompt, cond_embeddings, uncond_embeddings, merged_embeddings[-1], latents[-1], image_pos_np, image_neg_np, image_inv_np,lpips_score)
                
            i=i+1
            run[f'output/working'].log(f'LPIPS Score for {coco_id} is {lpips_score} and is less than 0.25, saved in dataset')
            if i!=0 and i%BATCH_SIZE==0:
                file.flush()
                file.close()
                print(f'Processed {i} prompts')
                DATA_FILE = f'{env.quality_dataset}/dataset_quality_{int((i/BATCH_SIZE))}.h5'
                print(f'New file name created {DATA_FILE}')
        else:
            print(f'LPIPS Score for {coco_id} is {lpips_score} and is greater than 0.25, not saving in dataset')
            run[f'output/notworking'].log(f'LPIPS Score for {coco_id} is {lpips_score} and is greater than 0.2, not saving in dataset')
    else:
        print(f'{coco_id} not in dataset')
        run[f'output/notworking'].log(f'{coco_id} not in dataset')