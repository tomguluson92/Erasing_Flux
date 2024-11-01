# coding: UTF-8
"""
    @date:  2024.10.28  week44 Monday
    @func:  ESD for Flux
"""
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ
from typing import Any, Callable, Dict, List, Optional, Union
from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from diffusers.utils.torch_utils import randn_tensor
import random
import glob
import re
import shutil
import pdb
import argparse

from diffusers import (
    FluxPipeline,
)

from utils import flux_pack_latents, flux_unpack_latents

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

@torch.no_grad()
def latent_sample(flux_model, batch_size, num_channels_latents, height, width, prompt_embeds, guidance, timesteps):
    """
        Sample the model
        ESD quick_sample_till_t
    """

    height = int(height) // 8  # self.vae_scale_factor
    width = int(width) // 8    # self.vae_scale_factor
    shape = (batch_size, num_channels_latents, height, width)
    
    # (A) generate random tensor
    latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, flux_model.device, torch.bfloat16)
    
    # (B) generate prompt embed
    prompt_embeds, pooled_prompt_embeds, text_ids = flux_model.encode_prompt(
                                                                    prompt=prompt,
                                                                    prompt_2=prompt,
                                                                    device=flux_model.device,
                                                                    num_images_per_prompt=1,
                                                                    max_sequence_length=256,
                                                                    )
    # (Pdb) prompt_embeds.shape
    # torch.Size([1, 256, 4096])
    # (Pdb) pooled_prompt_embeds.shape
    # torch.Size([1, 768])
    # import pdb; pdb.set_trace()

    # (C) generate latents w.r.t text embedding
    flux_model.scheduler.set_timesteps(timesteps, device=flux_model.device)
    timesteps = flux_model.scheduler.timesteps
    
    latents = latents.to(flux_model.device).bfloat16()
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    text_ids = text_ids.bfloat16()
    # Denoising loop
    for i, t in enumerate(timesteps):

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(torch.bfloat16)
        
        # import pdb; pdb.set_trace()
        # self.transformer.config.guidance_embeds False => guidance = None
        noise_pred = flux_model.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=None,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = flux_model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents, latent_image_ids


def predict_noise(model, latent_code, prompt_embeds, text_ids, latent_image_ids, pooled_prompt_embeds, guidance, timesteps, CPU_only=False):
    """
        ESD (apply_model)
    """
    
    if CPU_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:1")
        
    model.transformer = model.transformer.to(device)
    
    model_pred = model.transformer(
                    hidden_states=latent_code.to(device),
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep= (timesteps / 1000).to(device),
                    guidance=None,
                    pooled_projections=pooled_prompt_embeds.to(device),
                    encoder_hidden_states=prompt_embeds.to(device),
                    txt_ids=text_ids.to(device),
                    img_ids=latent_image_ids.to(device),
                    return_dict=False,
                )[0]
    
    print("20241028 predict noise e0 en ep", model_pred.device)
    
    model_pred = flux_unpack_latents(
        model_pred,
        height=512,
        width=512,
        vae_scale_factor=8,
    )

    return model_pred

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models():
    model_orig = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda:0")
    model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda:1")
    return model_orig, model

def train_esd(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, devices, seperator=None, image_size=512, ddim_steps=50):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)
    # MODEL TRAINING SETUP

    flux_model_org, flux_model = get_models()
    flux_model_org.vae.enable_slicing()
    flux_model_org.vae.enable_tiling()
    
    flux_model.vae.enable_slicing()
    flux_model.vae.enable_tiling()

    # TODO: choose parameters to train based on train_method
    parameters = []
    # aaa = [name for name, param in flux_model.transformer.named_parameters()]
    # with open("weights.txt", "a") as file:
    #     for item in aaa:
    #         file.writelines(item+"\n")
    # import pdb; pdb.set_trace()
    for name, param in flux_model.transformer.named_parameters():
        # train all layers except attns and time_embed layers
        if train_method == 'noattn':
            if 'attn' in name or 'time_text_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers in transformer_blocks
        if train_method == 'selfattn_one':
            if 'transformer_blocks' in name and 'attn' in name and 'single_transformer_blocks' not in name:
                print(name)
                parameters.append(param)
        # train only self attention layers in single_transformer_blocks
        if train_method == 'selfattn_two':
            if 'single_transformer_blocks' in name and 'attn' in name:
                print(name)
                parameters.append(param)
        # train only text attention layers
        if train_method == 'textattn':
            if 'add_k_proj' in name or 'add_q_proj' in name or 'add_v_proj' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not ('time_text_embed' in name):
                print(name)
                parameters.append(param)
    # set model to train
    flux_model.transformer.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    num_channels_latents = 16

    losses = []
    opt = torch.optim.AdamW(parameters, 
                            lr=lr, 
                            betas=(0.9, 0.99),
                            weight_decay=1e-04,
                            eps=1e-08)
    criteria = torch.nn.MSELoss()
    history = []

    name = f'Flux-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}'
    
    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words,1)[0]
        
        # get text embeddings for unconditional and conditional prompts
        emb_0, pooled_emb_0, text_ids_0 = flux_model_org.encode_prompt(prompt='',
                                               prompt_2='',
                                               device=flux_model_org.device,
                                               num_images_per_prompt=1,
                                               max_sequence_length=256,
                                               )
        emb_p, pooled_emb_p, text_ids_p = flux_model_org.encode_prompt(prompt=[word],
                                               prompt_2=[word],
                                               device=flux_model_org.device,
                                               num_images_per_prompt=1,
                                               max_sequence_length=256,
                                               )
        emb_n, pooled_emb_n, text_ids_n = flux_model_org.encode_prompt(prompt=[word],
                                               prompt_2=[word],
                                               device=flux_model_org.device,
                                               num_images_per_prompt=1,
                                               max_sequence_length=256,
                                               )

        opt.zero_grad()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
        # start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
        
        # print("time enc", t_enc)
        with torch.no_grad():
            # generate an image with the concept from ESD model
            z, latent_image_ids = latent_sample(flux_model, 
                                                1, 
                                                num_channels_latents, 
                                                512,
                                                512,
                                                emb_p.to(devices[0]), 
                                                start_guidance, 
                                                int(t_enc)) # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0 = predict_noise(flux_model_org, z, emb_0, text_ids_0, latent_image_ids, pooled_emb_0, guidance=None, timesteps=t_enc_ddpm.to(devices[1]), CPU_only=True)
            # flux_model_org.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
            e_p = predict_noise(flux_model_org, z, emb_p, text_ids_p, latent_image_ids, pooled_emb_p, guidance=None, timesteps=t_enc_ddpm.to(devices[1]), CPU_only=True)

        # get conditional score from ESD model
        e_n = predict_noise(flux_model, z, emb_n, text_ids_n, latent_image_ids, pooled_emb_n, guidance=None, timesteps=t_enc_ddpm.to(devices[1]), CPU_only=False)
        e_0.requires_grad = False
        e_p.requires_grad = False
        
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[1]), e_0.to(devices[1]) - (negative_guidance*(e_p.to(devices[1]) - e_0.to(devices[1])))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
        # save checkpoint and loss curve
        # if (i+1) % 500 == 0 and i+1 != iterations and i+1>= 500:
            # save_model(model, name)

        if i % 100 == 0:
            save_history(losses, name, word_print)

    # flux_model.transformer.eval()

    save_model(flux_model.transformer, name)
    save_history(losses, name, word_print)


def save_model(model, name):
    # SAVE MODEL

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)

    model_name = f'models/{name}_flux_transformer.pt'
    torch.save(model.state_dict(), model_name)
        

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD Flux',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=int, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,1')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps)