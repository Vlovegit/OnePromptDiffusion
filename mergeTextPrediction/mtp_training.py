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
from torch.optim.lr_scheduler import LambdaLR

from model import LearnedEmbeddingModel 
import random



scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ss.huggingface_token
LOW_RESOURCE = False 
NUM_DDIM_STEPS = env.NUM_DDIM_STEPS
GUIDANCE_SCALE = env.GUIDANCE_SCALE
MAX_NUM_WORDS = 77
device = torch.device(env.device) if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = SimpleDaamPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler, seed=2551055002497238).to(device)
learningmodel = LearnedEmbeddingModel(env.device)
learningmodel = learningmodel.to(device)
optimizer = Adam(learningmodel.parameters(), lr=1e-2)  # Adam optimizer with ith learning rate of 0.01
lambda1 = lambda epoch: 1 - epoch / 100
learnscheduler = LambdaLR(optimizer, lr_lambda=lambda1)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer

#Neptune Configuration
experimentName = f"MLP Model Training with learning rate {1e-2}"

if experimentName is not None:
    run = neptune.init_run(
        project=ss.neptune_project,
        api_token=ss.neptune_api_token,
        name=experimentName,
        tags=['Model Training']
    )
os.makedirs(env.object_model, exist_ok=True)
torch.save(learningmodel.state_dict(), f"{env.object_model}/mto_object_untrained.pth")
#Find the learned embeddings
dataset_folder = env.object_dataset

print(f'Processing files list: {os.listdir(dataset_folder)}')

early_stopping_patience = 7  # How many epochs to wait after last time validation loss improved.
best_val_loss = float('inf')
last_val_loss = float('inf')
patience_counter = 0
file_names = os.listdir(dataset_folder)

#Randomly Create a segment for train and val files
train_files = random.sample(file_names, int(0.8 * len(file_names)))
val_files = list(set(file_names) - set(train_files))

eval_folder = f"{env.object_dataset}_evaluation"
os.makedirs(eval_folder, exist_ok=True)

# Copy validation files to the evaluation folder
for file_name in val_files:
    source_file = os.path.join(dataset_folder, file_name)
    destination_file = os.path.join(eval_folder, file_name)
    shutil.copyfile(source_file, destination_file)

val_files = os.listdir(eval_folder)

print(f'Train Files: {train_files}')
print(f'Validation Files: {val_files}')

loss_values = {}
epochs = 10
for epoch in range(epochs):
    total_train_loss = 0
    i = 0
    learningmodel.train()
    
    # Shuffle the list of file names
    random.shuffle(train_files)

    for filename in train_files:
        file_path = os.path.join(dataset_folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.h5'):
            print(f'Processing file: {file_path}')
            try:
                with h5.File(file_path, 'r') as file:
                    keys = list(file.keys())
                    random.shuffle(keys)
                    # Access the dataset
                    for key in keys:
                        dset = file[key]
                        if key.endswith('_lt'):
                            continue
                        else:
                            merged_embeddings = torch.tensor(dset[0]["merged_embeddings"]).to(device)
                            cond_embeddings = torch.tensor(dset[0]["cond_embeddings"]).to(device)
                            uncond_embeddings = torch.tensor(dset[0]["uncond_embeddings"]).to(device)
                            prompt = dset[0]["prompt"]
                            negative_prompt = dset[0]["negative_prompt"]
                            coco_id = key
                            print(f'Processing Prompt: {prompt} Negative Prompt: {negative_prompt} with COCO id: {coco_id}')              
                            optimizer.zero_grad()
                            learned_embeddings = learningmodel(cond_embeddings, uncond_embeddings)
                            learned_embeddings = learned_embeddings.to(device)
                            loss = nnf.mse_loss(learned_embeddings, merged_embeddings)
                            loss.backward()
                            optimizer.step()
                            print(f'Loss: {loss}')
                            loss_value = loss.item()
                            total_train_loss += loss_value
                            if epoch == 0:
                                loss_values[coco_id] = str(loss_value)
                            else:
                                loss_values[coco_id] = loss_values.get(coco_id) +'|'+ str(loss_value)
                            run[f'epoch_{epoch+1}/Train Loss'].log(loss_value)
                            i += 1
            except Exception as e:
                print(f'Error processing file: {file_path}')
                print(f'Error: {e}')
                continue

    total_val_loss = 0
    j = 0
    learningmodel.eval()
    with torch.no_grad():
        for filename in val_files:
            file_path = os.path.join(eval_folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.h5'):
                print(f'Processing file: {file_path}')
                try:
                    with h5.File(file_path, 'r') as file:
                        # Access the dataset
                        for key in file.keys():
                            dset = file[key]
                            if key.endswith('_lt'):
                                continue
                            else:
                                merged_embeddings = torch.tensor(dset[0]["merged_embeddings"]).to(device)
                                cond_embeddings = torch.tensor(dset[0]["cond_embeddings"]).to(device)
                                uncond_embeddings = torch.tensor(dset[0]["uncond_embeddings"]).to(device)
                                prompt = dset[0]["prompt"]
                                negative_prompt = dset[0]["negative_prompt"]
                                coco_id = key
                                print(f'Processing Prompt: {prompt} Negative Prompt: {negative_prompt} with COCO id: {coco_id}')             
                                optimizer.zero_grad()
                                learned_embeddings = learningmodel(cond_embeddings, uncond_embeddings)
                                learned_embeddings = learned_embeddings.to(device)
                                loss = nnf.mse_loss(learned_embeddings, merged_embeddings)
                                print(f'Loss: {loss}')
                                loss_value = loss.item()
                                total_val_loss += loss_value
                                if epoch == 0:
                                    loss_values[coco_id] = str(loss_value)
                                else:
                                    loss_values[coco_id] = loss_values.get(coco_id) +'|'+ str(loss_value)
                                run[f'epoch_{epoch+1}/Val Loss'].log(loss_value)
                                j += 1
                except Exception as e:
                    print(f'Error processing file: {file_path}')
                    print(f'Error: {e}')
                    continue

    # Update the learning rate
    learnscheduler.step()

    # You can check the current learning rate with
    current_lr = learnscheduler.get_last_lr()[0]
    print(f'Epoch: {epoch+1}, Current learning rate: {current_lr}')
    run[f'Learning Rate'].log(f'{epoch + 1} Learning rate is : {current_lr}')

    # Calculate your epoch_loss here
    mean_train_loss = total_train_loss / i
    mean_val_loss = total_val_loss / j

    # Log the epoch loss
    run[f'mean_loss/Train'].log(mean_train_loss)  # Log mean train loss
    run[f'mean_loss/Validation'].log(mean_val_loss)  # Log mean validation loss

    if epoch == 0:
        best_val_loss = mean_val_loss
        last_val_loss = mean_val_loss
        torch.save(learningmodel.state_dict(), f"{env.object_model}/mto_object_{epoch}_trained.pth") # Save the first model
    elif mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        last_val_loss = mean_val_loss
        torch.save(learningmodel.state_dict(), f"{env.object_model}/mto_object_{epoch}_trained.pth")  # Save the best model
        patience_counter = 0  # Reset counter
    elif mean_val_loss < last_val_loss:
        last_val_loss = mean_val_loss
        patience_counter = 0
    else:
        last_val_loss = mean_val_loss
        patience_counter += 1  # Increment counter
    
    if patience_counter >= early_stopping_patience:
        print(f'Stopping early at epoch {epoch+1}')
        break

    # Save the loss values to csv file
    with open(f'{env.object_model}/loss_values.csv', 'w') as f:
        for key in loss_values.keys():
            f.write("%s,%s\n"%(key,loss_values[key]))

# #Delete the evaluation files
# for filename in val_files:
#     file_path = os.path.join(eval_folder, filename)
#     os.remove(file_path)


torch.save(learningmodel.state_dict(), f"{env.object_model}/mto_object_trained.pth")
#Log the train and val files list to neptune
print(f'Train Files: {train_files}')
print(f'Validation Files: {val_files}')
run['Train Files'].log(train_files)
run['Validation Files'].log(val_files)
# Stop the Neptune run
run.stop()