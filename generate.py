from argparse import ArgumentParser
from stable_diffusion import StableDiffusion



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate image by text prompts using Stable Diffusion")
    parser.add_argument("text_prompts", nargs="+", type=str, default=None,
                        help="Text prompts for image generation")
    parser.add_argument("--scheduler_name", type=str, default="pndm",
                        help="Scheduler name that be used during inference. Default: 'pndm'")
    parser.add_argument("--device", type=str, default=None,
                        help="Name of the device that be used during inference. Default: None")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    StableDiffusion(scheduler_name=args.scheduler_name, 
                    device=args.device).generate(args.text_prompts)