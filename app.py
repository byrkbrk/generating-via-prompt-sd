from stable_diffusion import StableDiffusion
import gradio as gr



if __name__ == "__main__":
    stable_diffusion = StableDiffusion(create_dirs=False)
    gr_interface = gr.Interface(
        fn=lambda prompts, save=False, show=False: stable_diffusion.generate(prompts.split("\n"),
                                                                             save,
                                                                             show)[0],
        inputs=[gr.Textbox(lines=3, placeholder="an image of a turtle in Camille Pissarro style")],
        outputs=gr.Image(type="pil"),
        title="Stable Diffusion-v1-5"
    )
    gr_interface.launch()




