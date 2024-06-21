import torch
import sys
 
sys.path.insert(1, '/home/myid/vg80700/gits/OnePromptDiffusion/utils')
sys.path.insert(2, '/home/myid/vg80700/gits/OnePromptDiffusion/mergeTextOptimization')

from pipes import SimpleDaamPipeline
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch.nn.functional as nnf
import abc
from ptp_utils import *
import shutil
from torch.optim.adam import Adam
import supersecrets as ss
import neptune
import os
import h5py as h5
import config as env
from model import LearnedEmbeddingModel 
from modified_nto import MergeTextOptimization as mto
import pandas as pd
from PIL import Image
from neptune.types import File

# Load the model
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ss.huggingface_token
LOW_RESOURCE = False 
NUM_DDIM_STEPS = env.NUM_DDIM_STEPS
GUIDANCE_SCALE = env.GUIDANCE_SCALE
MAX_NUM_WORDS = 77
device = torch.device(env.device) if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = SimpleDaamPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler, seed=2551055002497238).to(device)
mto_inversion = mto(ldm_stable, env.NUM_DDIM_STEPS, env.GUIDANCE_SCALE, env.device)

model_path_untrained = f"{env.object_model}/mto_object_untrained.pth"
model_path_trained = f"{env.object_model}/mto_object_trained.pth"


# For the untrained model
learningmodeluntrained = LearnedEmbeddingModel(env.device)
state_dict_untrained = torch.load(model_path_untrained, map_location=env.device) # Correctly loading from path
learningmodeluntrained.load_state_dict(state_dict_untrained)
learningmodeluntrained.to(device)
learningmodeluntrained.eval()

# For the model with the best training performance
learningmodeltrainedbest = LearnedEmbeddingModel(env.device)
state_dict_trained_best = torch.load(model_path_trained, map_location=env.device) # Correctly loading from path
learningmodeltrainedbest.load_state_dict(state_dict_trained_best)
learningmodeltrainedbest.to(device)
learningmodeltrainedbest.eval()

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

#Neptune Configuration
experimentName = "MLP Model Evaluation"

if experimentName is not None:
    run = neptune.init_run(
        project=ss.neptune_project,
        api_token=ss.neptune_api_token,
        name=experimentName,
        tags=['empty_dataset']
    )

# Initialize list to store score differences along with COCO IDs
lpips_differences = []
fid_differences = []
is_differences = []

# Initialize list to store best-trained LPIPS scores along with COCO IDs
best_trained_lpips_scores = []
best_trained_fid_scores = []
best_trained_is_scores = []

#Find the learned embeddings
dataset_folder = f"{env.object_dataset}_evaluation"
os.makedirs(f"{env.fid_metric}/fid_ut", exist_ok=True)
os.makedirs(f"{env.fid_metric}/fid_best", exist_ok=True)
os.makedirs(f"{env.fid_metric}/fid_neg", exist_ok=True)

for filename in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, filename)
    if os.path.isfile(file_path) and filename.endswith('.h5'):
        print(f'Processing file: {file_path}')
        with h5.File(file_path, 'r') as file:
            i = 0
            # Access the dataset
            for key in file.keys():
                dset = file[key]
                if key.endswith('_lt'):
                    continue
                else:
                    latent = torch.tensor(dset[0]["latents"]).to(device)
                    cond_embeddings = torch.tensor(dset[0]["cond_embeddings"]).to(device)
                    uncond_embeddings = torch.tensor(dset[0]["uncond_embeddings"]).to(device)
                    merged_embeddings_gt = torch.tensor(dset[0]["merged_embeddings"]).to(device)
                    prompt = dset[0]["prompt"]
                    negative_prompt = dset[0]["negative_prompt"]
                    coco_id = key
                    image_pos = dset[0]["image_pos"]
                    image_neg = dset[0]["image_neg"]
                    image_inv = dset[0]["image_inv"]

                    print(f'Processing file: {file_path}')
                    print(f'Processing Prompt: {prompt} Negative Prompt: {negative_prompt} with COCO id: {coco_id}')

                    merged_embeddings_untrained = learningmodeluntrained(cond_embeddings, uncond_embeddings)
                    merged_embeddings_trained_best = learningmodeltrainedbest(cond_embeddings, uncond_embeddings)
                    image_inv_ut, x_t1 = mto_inversion.run_and_display_merged(merged_embeddings_untrained, ldm_stable, run_baseline=False, latent=latent, verbose=False)
                    image_inv_best, x_t1 = mto_inversion.run_and_display_merged(merged_embeddings_trained_best, ldm_stable, run_baseline=False, latent=latent, verbose=False)
                    
                    # Compute LPIPS scores
                    lpips_score_ut = compute_lpips(image_inv_ut[0], image_neg)
                    lpips_score_ut = round(lpips_score_ut, 4)
                    lpips_score_best = compute_lpips(image_inv_best[0], image_neg)
                    lpips_score_best = round(lpips_score_best, 4)
                    lpips_difference = lpips_score_ut - lpips_score_best
                    lpips_differences.append((key, lpips_difference, lpips_score_ut, lpips_score_best))
                    best_trained_lpips_scores.append((key, lpips_score_best))

                    # Save images                    
                    image_neg_path = f"{env.fid_metric}/fid_neg"
                    image_best_path = f"{env.fid_metric}/fid_best"
                    image_ut_path = f"{env.fid_metric}/fid_ut"
                    
                    image_inv_ut = Image.fromarray(image_inv_ut[0].astype('uint8'), 'RGB')
                    image_inv_ut.save(f'{image_ut_path}/image_inv_ut.jpg')
                    image_inv_best = Image.fromarray(image_inv_best[0].astype('uint8'), 'RGB')
                    image_inv_best.save(f'{image_best_path}/image_inv_best.jpg')
                    image_neg = Image.fromarray(image_neg, 'RGB')
                    image_pos = Image.fromarray(image_pos, 'RGB')
                    image_inv = Image.fromarray(image_inv, 'RGB')
                    image_neg.save(f'{image_neg_path}/image_gt.jpg')
                    # Compute FID, IS and Semantic Similarity                   
                    fid_score_gt_ut = compute_fid(image_neg_path, image_ut_path)
                    fid_score_gt_best = compute_fid(image_neg_path, image_best_path)
                    if fid_score_gt_ut is not None and fid_score_gt_best is not None:
                        fid_score_gt_ut = round(float(fid_score_gt_ut), 4)
                        fid_score_gt_best = round(float(fid_score_gt_best), 4)
                        fid_difference = round((fid_score_gt_ut - fid_score_gt_best),4)
                        fid_differences.append((key, fid_difference, fid_score_gt_ut, fid_score_gt_best))
                        best_trained_fid_scores.append((key, fid_score_gt_best))
                        run[f'output/images/{coco_id}/metrics'].log(f'FID Score Untrained : {fid_score_gt_ut}')
                        run[f'output/images/{coco_id}/metrics'].log(f'FID Score Best Trained : {fid_score_gt_best}')
                        run[f'output/images/{coco_id}/metrics'].log(f'FID Difference : {fid_difference}')
                    
                    #Delete saved images
                    os.remove(f'{image_ut_path}/image_inv_ut.jpg')
                    os.remove(f'{image_best_path}/image_inv_best.jpg')
                    os.remove(f'{image_neg_path}/image_gt.jpg')

                    run[f'output/images/{coco_id}/images'].append(image_pos,description=f'Image with Positive Prompt Only Ground Truth : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
                    run[f'output/images/{coco_id}/images'].append(image_neg,description=f'Image with Negative Prompt Ground Truth : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
                    run[f'output/images/{coco_id}/images'].append(File.as_image(image_inv),description=f'Image Inversion NTO : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
                    run[f'output/images/{coco_id}/images'].append(File.as_image(image_inv_ut),description=f'Image Inversion Untrained : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
                    # run[f'output/images/{coco_id}/images'].append(File.as_image(image_inv_t),description=f'Image Inversion Last Trained : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')
                    run[f'output/images/{coco_id}/images'].append(File.as_image(image_inv_best),description=f'Image Inversion Best Trained : for \n prompt:{prompt}\n np: {negative_prompt}\n COCO id: {coco_id}')

                    run[f'output/images/{coco_id}/metrics'].log(f'LPIPS Score Untrained : {lpips_score_ut}')
                    # run[f'output/images/{coco_id}/metrics'].log(f'LPIPS Score Last Trained : {lpips_score_lt}')
                    run[f'output/images/{coco_id}/metrics'].log(f'LPIPS Score Best Trained : {lpips_score_best}')
                    run[f'output/images/{coco_id}/metrics'].log(f'LPIPS Difference : {lpips_difference}')
   
# Sort by metrics to find top and bottom performers
lpips_differences.sort(key=lambda x: x[1], reverse=True)
fid_differences.sort(key=lambda x: x[1], reverse=True)

# Extract top 50 best and worst performers
top_100_best_lpips = lpips_differences[:100]
top_100_worst_lpips = lpips_differences[-100:]
top_100_best_fid = fid_differences[:100]
top_100_worst_fid = fid_differences[-100:]

# Log the top 50 best and worst performers on Neptune
for rank, (coco_id, diff, score_ut, score_best) in enumerate(top_100_best_lpips, 1):
    run[f'output/analysis/top_100_best'].append(f'Rank: {rank}, COCO ID: {coco_id}, LPIPS Difference: {diff}, Untrained: {score_ut}, Best Trained: {score_best}')

for rank, (coco_id, diff, score_ut, score_best) in enumerate(reversed(top_100_worst_lpips), 1):
    run[f'output/analysis/top_100_worst'].append(f'Rank: {rank}, COCO ID: {coco_id}, LPIPS Difference: {diff}, Untrained: {score_ut}, Best Trained: {score_best}')

for rank, (coco_id, diff, score_ut, score_best) in enumerate(top_100_best_fid, 1):
    run[f'output/analysis/top_100_best_fid'].append(f'Rank: {rank}, COCO ID: {coco_id}, FID Difference: {diff}, Untrained: {score_ut}, Best Trained: {score_best}')

for rank, (coco_id, diff, score_ut, score_best) in enumerate(reversed(top_100_worst_fid), 1):
    run[f'output/analysis/top_100_worst_fid'].append(f'Rank: {rank}, COCO ID: {coco_id}, FID Difference: {diff}, Untrained: {score_ut}, Best Trained: {score_best}')

#Log all the lpips differences
for coco_id, diff, score_ut, score_best in lpips_differences:
    run[f'output/analysis/all_lpips_differences'].append(f'COCO ID: {coco_id}, LPIPS Difference: {diff}, Untrained: {score_ut}, Best Trained: {score_best}')

#Log all the fid differences
for coco_id, diff, score_ut, score_best in fid_differences:
    run[f'output/analysis/all_fid_differences'].append(f'COCO ID: {coco_id}, FID Difference: {diff}, Untrained: {score_ut}, Best Trained: {score_best}')

# Sort by best-trained LPIPS scores to find the entries with minimum scores
best_trained_lpips_scores.sort(key=lambda x: x[1])
best_trained_fid_scores.sort(key=lambda x: x[1])

# Extract top 50 entries with the minimum LPIPS scores
top_100_minimum_lpips_scores = best_trained_lpips_scores[:100]
top_100_minimum_fid_scores = best_trained_fid_scores[:100]



# Log the top 50 entries with the minimum best-trained LPIPS scores on Neptune
for rank, (coco_id, score_best) in enumerate(top_100_minimum_lpips_scores, 1):
    run[f'output/analysis/top_100_minimum_scores'].log(f'Rank: {rank}, COCO ID: {coco_id}, Best Trained LPIPS Score: {score_best}')

for rank, (coco_id, score_best) in enumerate(top_100_minimum_fid_scores, 1):
    run[f'output/analysis/top_100_minimum_fid_scores'].log(f'Rank: {rank}, COCO ID: {coco_id}, Best Trained FID Score: {score_best}')

#Save all the values to the excel sheet

# Step 1: Organize your data
data_best_lpips = [{'Rank': rank, 'COCO ID': coco_id, 'LPIPS Difference': diff, 'Untrained': score_ut, 'Best Trained': score_best}
                   for rank, (coco_id, diff, score_ut, score_best) in enumerate(top_100_best_lpips, 1)]
data_worst_lpips = [{'Rank': rank, 'COCO ID': coco_id, 'LPIPS Difference': diff, 'Untrained': score_ut, 'Best Trained': score_best}
                    for rank, (coco_id, diff, score_ut, score_best) in enumerate(reversed(top_100_worst_lpips), 1)]
data_best_fid = [{'Rank': rank, 'COCO ID': coco_id, 'FID Difference': diff, 'Untrained': score_ut, 'Best Trained': score_best}
                 for rank, (coco_id, diff, score_ut, score_best) in enumerate(top_100_best_fid, 1)]
data_worst_fid = [{'Rank': rank, 'COCO ID': coco_id, 'FID Difference': diff, 'Untrained': score_ut, 'Best Trained': score_best}
                  for rank, (coco_id, diff, score_ut, score_best) in enumerate(reversed(top_100_worst_fid), 1)]
data_minimum_lpips_scores = [{'Rank': rank, 'COCO ID': coco_id, 'Best Trained LPIPS Score': score_best}
                       for rank, (coco_id, score_best) in enumerate(top_100_minimum_lpips_scores, 1)]
data__lpips_metrics = [{'COCO ID': coco_id, 'LPIPS Difference': diff, 'Untrained LPIPS': score_ut, 'Best Trained LPIPS': score_best}
                            for coco_id, diff, score_ut, score_best in lpips_differences]
data_fid_metrics = [{'COCO ID': coco_id, 'FID Difference': diff, 'Untrained FID': score_ut, 'Best Trained FID': score_best}
                            for coco_id, diff, score_ut, score_best in fid_differences]
data_minimum_fid_scores = [{'Rank': rank, 'COCO ID': coco_id, 'Best Trained FID Score': score_best}
                          for rank, (coco_id, score_best) in enumerate(top_100_minimum_fid_scores, 1)]

# Convert lists to dictionaries with COCO ID as key for easy lookup
lpips_dict = {item['COCO ID']: item for item in data__lpips_metrics}
fid_dict = {item['COCO ID']: item for item in data_fid_metrics}

# Initialize a new list for combined data
combined_data = []

# Merge the dictionaries
for coco_id, lpips_metrics in lpips_dict.items():
    combined_metrics = lpips_metrics  # Start with LPIPS metrics
    
    # If the same COCO ID exists in FID metrics, combine the data
    if coco_id in fid_dict:
        combined_metrics.update(fid_dict[coco_id])  # Update combines the two dictionaries

    combined_data.append(combined_metrics)

# For FID metrics not in LPIPS (if any)
for coco_id, fid_metrics in fid_dict.items():
    if coco_id not in lpips_dict:
        combined_data.append(fid_metrics)


# Step 2: Create pandas DataFrames
df_best_lpips = pd.DataFrame(data_best_lpips)
df_worst_lpips = pd.DataFrame(data_worst_lpips)
df_best_fid = pd.DataFrame(data_best_fid)
df_worst_fid = pd.DataFrame(data_worst_fid)
df_minimum_lpips_scores = pd.DataFrame(data_minimum_lpips_scores)
df_minimum_fid_scores = pd.DataFrame(data_minimum_fid_scores)
df_metric_scores = pd.DataFrame(combined_data)


# Step 3: Use pandas ExcelWriter to save all DataFrames
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(env.performance_analysis, exist_ok=True)
filename = f'{env.performance_analysis}/object_{timestamp}.xlsx'
with pd.ExcelWriter(filename) as writer:
    df_best_lpips.to_excel(writer, sheet_name='Top 100 Best LPIPS', index=False)
    df_worst_lpips.to_excel(writer, sheet_name='Top 100 Worst LPIPS', index=False)
    df_best_fid.to_excel(writer, sheet_name='Top 100 Best FID', index=False)
    df_worst_fid.to_excel(writer, sheet_name='Top 100 Worst FID', index=False)
    df_minimum_lpips_scores.to_excel(writer, sheet_name='Top 100 Minimum LPIPS Scores', index=False)
    df_minimum_fid_scores.to_excel(writer, sheet_name='Top 100 Minimum FID Scores', index=False)
    df_metric_scores.to_excel(writer, sheet_name='Metric Scores', index=False)

print(f'Data saved to {filename}')

run.stop()                    