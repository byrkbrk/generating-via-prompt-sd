# Generating Image via Text Prompts using SD

## Introduction

We implement a module that generates images based on the user-defined (text) prompts. While preparing the module, we utilized the pretrained model Stable Diffusion v1-5 provided by [runwayml in HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5).

## Setting Up the Environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository
    ~~~
    git clone https://github.com/byrkbrk/generating-via-prompt-sd.git
    ~~~
3. Change the directory:
    ~~~
    cd generating-via-prompt-sd
    ~~~
4. For macos, run:
    ~~~
    conda env create -f generating-via-prompt_macos.yaml
    ~~~
    For linux or windows, run:
    ~~~
    conda env create -f generating-via-prompt_linux.yaml
    ~~~
5. Activate the environment:
    ~~~
    conda activate generating-via-prompt-sd
    ~~~

## Generating Images