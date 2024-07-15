import os
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler



class StableDiffusion(object):
    def __init__(self, 
                 ckpt_name="runwayml/stable-diffusion-v1-5",
                 scheduler_name="pndm",
                 device=None,
                 create_dirs=True):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.pipeline = self.instantiate_pipeline(ckpt_name, scheduler_name, self.device)
        if create_dirs: self.create_dirs(self.module_dir)
        
    def generate(self, prompts, save=True, show=True):
        "Returns the list of generated images based on given text prompts"
        images = self.pipeline(prompts).images
        for i, image in enumerate(images):
            if save: 
                image.save(os.path.join(self.module_dir, 
                                        "generated-images", 
                                        f"generated_image_prompt_{i}.jpg"))
            if show: image.show()
        return images
    
    def instantiate_pipeline(self, ckpt_name, scheduler_name, device):
        """Returns instantiated diffusion pipeline based on the given arguments"""
        pipeline = DiffusionPipeline.from_pretrained(ckpt_name, use_safetensors=True).to(device)
        if scheduler_name == "pndm":
            pass
        elif scheduler_name == "euler":
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
    
    def create_dirs(self, root):
        """Creates the required directories under given root directory"""
        dir_names = ["generated-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)

