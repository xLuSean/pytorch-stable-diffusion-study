# ref : https://edge.aif.tw/express-stable-diffusion/
from dotenv import load_dotenv
import os
import torch

load_dotenv()

device = os.getenv("DEVICE", default="cpu")
print(device)

prompt = ["RAW Photography,Snow-capped mountains, flowers and moos, sunrise,  sunrays, white clouds, lens flare, low wide angle, Canon EOS 5D Mark IV, masterpiece, 35mm photograph, film grain, award winning photography, vibrant use of light and shadow, vivid colors, high quality textures of materials, volumetric textures  perfect composition, dynamic play of light, rich colors, epic shot, perfectly quality, natural textures, high detail, high sharpness, high clarity, detailed ,photoshadow,  intricate details, 8k"]

height = 512
width = 512
batch_size = 1
generator = torch.manual_seed(42)

# we are generating 64x64 images
latents = torch.randn((batch_size, 4, height //8, width //8), generator = generator, )
latents = latents.to(device)

print(latents.shape)

# ========================================================================================

from transformers import CLIPTextModel, CLIPTokenizer

# text embedding model and tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)

# ========================================================================================

text_encoder = text_encoder.to(device)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt") # tokenizing the prompt

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0] # token to embedding

print(text_embeddings.shape)

# 在Stable Diffusion的生成中，classifier-free guidance是個要特別注意的概念。因為，如果我們都以文字控制生成圖片的樣式，可能會太過侷限。因此，我們會期待這些生成的圖片具有變化性，例如背景能帶點自己的想法或是具設計感的汽車造型，而這些就需要輸入無指涉的文字，例如將空白字符當成文字，這樣就不會有文字意義了。

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

print(uncond_embeddings.shape)

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
print(text_embeddings.shape)

# ========================================================================================

from diffusers import UNet2DConditionModel, LMSDiscreteScheduler

unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, subfolder = "unet")

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, subfolder = "scheduler")

# ========================================================================================

from tqdm.auto import tqdm
from torch import autocast

num_inference_steps = 20    # number of denoising steps, 50 is recommended
guidance_scale = 7.5        # classifier-free guidance, 7.5 is recommended

unet = unet.to(device)
scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

scheduler.timesteps = scheduler.timesteps.type(torch.float32) # for apple silicon

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # print(latent_model_input.shape)
    # print(text_embeddings.shape)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states = text_embeddings).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = scheduler.step(noise_pred, t, latents).prev_sample

# ========================================================================================
from diffusers import AutoencoderKL

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, subfolder = "vae")

import matplotlib.pyplot as plt

vae = vae.to(device)

latents = 1/ 0.18215 * latents # scale and decode the image latents with vae

with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0,2,3,1).numpy()   # to cpu
images = (image*255).round().astype("uint8")
plt.imshow(images[0])
plt.show()  # This actually displays the image

# Save the first image
plt.imsave("output_image.png", images[0])