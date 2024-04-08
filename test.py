from diffusers import StableDiffusionXLPipeline
import torch
import time

start_time = time.perf_counter()

## Load the pipeline in full-precision and place its model components on CUDA.
# on apple silicon, using "mps" is recommended for better performance.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
    ).to("cpu")

## Run the attention ops without efficiency.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=5).images[0]
image.save("astronaut_in_jungle.png")

end_time = time.perf_counter()
print(f"Time taken: {end_time - start_time:.2f} seconds")