import os
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler



class StableDiffusion(object):
    def __init__(self, 
                 ckpt_name="runwayml/stable-diffusion-v1-5",
                 scheduler_name=None,
                 device=None):
        self.device = self.initialize_device(device)
        self.pipeline = self.instantiate_pipeline(ckpt_name, scheduler_name, self.device)
    
    def generate(self, prompts, save=True, show=True):
        "Returns the list of generated images based on given text prompts"
        images = self.pipeline(prompts).images
        for i, image in enumerate(images):
            if save: image.save(f"generated_image_prompt_{i}.jpg")
            if show: image.show()
        return images
    
    def instantiate_pipeline(self, ckpt_name, scheduler_name, device):
        """Returns instantiated diffusion pipeline based on the given arguments"""
        pipeline = DiffusionPipeline.from_pretrained(ckpt_name, use_safetensors=True).to(device)
        if scheduler_name == "euler":
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        return pipeline
    
    def initialize_device(self, device):
        """Returns device based on GPU availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
            


if __name__ == "__main__":
    stable_diffusion = StableDiffusion(scheduler_name="euler")
    #print(stable_diffusion.pipeline)
    #prompt = "An image of Johann Sebastian Bach while composing an opera on his wood table at night"
    #prompt = "An image of Mini Cooper on the roda, at a rainy night, in realistic style"
    #prompt = "a photo of an astronaut riding a horse on mars"
    #prompt = "a realistic photo of an Italian woman, 4K, colorful, wearing hat"
    #prompt = "an image of a lion in Monet style"
    #stable_diffusion.generate(prompt)
    prompts = ["an image of a lion in Monet style", "an image of a lion in Picasso style"]
    stable_diffusion.generate(prompts)

    

