import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv
import os
import torch
import time

load_dotenv()
device = os.getenv("DEVICE", default="cpu")

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

start_time = time.perf_counter()

# Load model.
# unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
# unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = StableDiffusionXLPipeline.from_pretrained(base,  torch_dtype=torch.float16, variant="fp16").to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")

end_time = time.perf_counter()

print(f"Time taken: {end_time - start_time:.2f}s")