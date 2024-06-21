from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
import sys
 
sys.path.insert(1, '/home/myid/vg80700/gits/OnePromptDiffusion/utils')

from typing import Optional, Union
from tqdm.notebook import tqdm
import torch
from diffusers import DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
from ptp_utils import *
from torch.optim.adam import Adam
from PIL import Image


class MergeTextOptimization:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = latents
        if context is None:
            context = self.context
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str, neg_prompt: str):
        uncond_input = self.model.tokenizer(
            [neg_prompt], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        learned_embeddings = torch.randn(cond_embeddings.shape) * 0.01
        learned_embeddings = learned_embeddings.to(self.device)
        
        learned_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.NUM_DDIM_STEPS)
        for i in range(self.NUM_DDIM_STEPS):
            learned_embeddings = learned_embeddings.clone().detach()
            learned_embeddings.requires_grad = True
            optimizer = Adam([learned_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
               
            for j in range(num_inner_steps):
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, learned_embeddings)
                latents_prev_rec = self.prev_step(noise_pred_cond, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            learned_embeddings_list.append(learned_embeddings[:1].detach())
            with torch.no_grad():
                latent_cur = self.get_noise_pred(latent_cur, t, False, learned_embeddings)
        bar.close()
        return learned_embeddings_list, uncond_embeddings, cond_embeddings
    
    def invert(self, image_path: str, prompt: str, negative_prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt,negative_prompt)
        image_gt = self.load_512_mod(image_path)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        learned_embeddings, uncond_embeddings, cond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1],learned_embeddings, cond_embeddings, uncond_embeddings, ddim_latents
    
    def load_512_mod(self, image_path):
        if type(image_path) is str:
            image = np.array(Image.open(image_path))[:, :, :3]
        else:
            image = image_path

        image = np.array(Image.fromarray(image).resize((512, 512)))
        return image
    
    @torch.no_grad()
    def text2image_ldm_stable_merged(
        self,
        model,
        learned_embedding: None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        return_type='image'
    ):
        height = width = 512
        batch_size = 1

        latent, latents = init_latent(latent, model, height, width, generator, batch_size)
        model.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
            latents = diffusion_step_mod(model, latents, learned_embedding, t, guidance_scale, low_resource=False)
            
        if return_type == 'image':
            image = latent2image(model.vae, latents)
        else:
            image = latents
        return image, latent

    def run_and_display_merged(self,learned_embedding, ldm_stable, latent=None, run_baseline=False, generator=None, verbose=True):
        if run_baseline:
            print("w.o. prompt-to-prompt")
            images, latent = self.run_and_display_merged(learned_embedding, latent=latent, run_baseline=False, generator=generator)
            print("with prompt-to-prompt")
        images, x_t = self.text2image_ldm_stable_merged(ldm_stable, learned_embedding, latent=latent, num_inference_steps=self.NUM_DDIM_STEPS, guidance_scale=self.GUIDANCE_SCALE, generator=generator)
        if verbose:
            view_images(images)
        return images, x_t
        
    
    def __init__(self, model, NUM_DDIM_STEPS, GUIDANCE_SCALE, DEVICE):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.GUIDANCE_SCALE = GUIDANCE_SCALE
        self.device = DEVICE

    