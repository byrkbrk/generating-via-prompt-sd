import os
from diffusers import DiffusionPipeline



class StableDiffusion(object):
    def __init__(self, ckpt_name="runwayml/stable-diffusion-v1-5"):
        self.device = "mps"
        self.pipeline = DiffusionPipeline.from_pretrained(ckpt_name, use_safetensors=True).to(self.device)
    
    def generate(self, prompt):
        "Returns the generated image based on given text prompt"
        image = self.pipeline(prompt).images[0]
        image.show()
        image.save("generated_image.jpg")
        return image



if __name__ == "__main__":
    stable_diffusion = StableDiffusion()
    print(stable_diffusion.pipeline)
    #prompt = "An image of Johann Sebastian Bach while composing an opera on his wood table at night"
    #prompt = "An image of Mini Cooper on the roda, at a rainy night, in realistic style"
    #prompt = "a photo of an astronaut riding a horse on mars"
    #prompt = "a realistic photo of an Italian woman, 4K, colorful, wearing hat"
    prompt = "an image of a lion in Monet style"
    stable_diffusion.generate(prompt)


