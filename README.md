# One Prompt Diffusion

![teaser_final](https://github.com/Vlovegit/OnePromptDiffusion/assets/22128055/7efa056b-eb7f-473d-a665-0ff8a6c9d14c)

## Abstract

Diffusion models have significantly advanced image synthesis, excelling in visual quality and flexibility. Traditional models use additional negative prompts with CFG to guide the generation process. However, CFG requires the model to run twice, complicating the interpretation of the negative prompt's impact on the final image. This research proposes a method to generate a single prompt that produces images of comparable quality to the two-prompt CFG approach, achieving up to 2x speedup and 20% memory reduction.

## Contributions

#### Single Prompt Generation: 
A novel approach to replace the multi-prompt CFG in diffusion models with a single prompt without losing visual quality in the generated images.

#### Prompt-to-Image Dataset: 
A comprehensive dataset with complex prompts and various negative prompt settings for object removal and quality improvement, including inversion latents and embeddings.

## Methodology

1. **Dataset Preparation**: Constructed a prompt-to-image dataset and performed per-image optimization to find the unified prompt.
2. **Merged-Text Optimization (MTO)**: Optimized the merged prompt to achieve the desired image quality.
3. **Merged-Text Prediction (MTP)**: Trained a neural network module to predict the embedding of the unified prompt from the input prompts.

## Results

- **Efficiency**: Achieved up to 2x speedup and 20% memory reduction compared to traditional CFG methods.
- **Image Quality**: Maintained high visual quality, comparable to CFG-enabled diffusion models.

## Repository Structure

- `code/`: Contains the implementation of the One Prompt Diffusion model.
- `data/`: Contains the prompt-to-image dataset and associated files.
- `results/`: Contains the results of the experiments, including images and performance metrics.
- `docs/`: Contains documentation and additional resources related to the project.

## Usage

1. **Installation**: Instructions for setting up the environment and installing dependencies.
2. **Running Experiments**: Steps to reproduce the experiments and generate images using the One Prompt Diffusion model.
3. **Evaluation**: Scripts to evaluate the performance of the model in terms of efficiency and image quality.

## Acknowledgments

This research was conducted under the guidance of Professor Jin Sun. Special thanks to the committee members, Professor Lakshmish Ramaswamy and Professor Wei Niu, for their support and encouragement. Gratitude is also extended to the University of Georgia for providing the resources and financial support necessary for this study.

## Contact

For any inquiries or further information, please contact Vaibhav Goyal at vaibhavgoyal17796@gmail.com.
