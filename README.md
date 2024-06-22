# One Prompt Diffusion

Diffusion models have significantly advanced image synthesis, excelling in visual quality and flexibility. Traditional models use additional negative prompts with CFG to guide the generation process. However, CFG requires the model to run twice, complicating the interpretation of the negative prompt's impact on the final image. This research proposes a method to generate a single prompt that produces images of comparable quality to the two-prompt CFG approach, achieving up to 2x speedup and 20% memory reduction.

![teaser_final](https://github.com/Vlovegit/OnePromptDiffusion/assets/22128055/7efa056b-eb7f-473d-a665-0ff8a6c9d14c)

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

- `mergeTextOptimization/`: Implementation to find the optimized single merged embeddings using a modified Null-text inversion algorithm (Ref: https://null-text-inversion.github.io/).
- `mergeTextPrediction/`: Implementation of training and evaluating MLP model to predict merged embeddings without optimization using significantly less memory and time than optimized embeddings
- `promptImageDataset/`: Scripts to generate dataset through multi-step filtering for training and evaluating the MLP model and is not limited to this research.
- `utils/`: Miscellaneous methods used across code. This folder contains code logic for a customized stable diffusion pipeline without using CFG and loading the coco dataset files.

## Usage

1. **Installation**:
- Clone the responsibility to your local machine
- Install the necessary libraries using the command `pip install -r requirements.txt`
- Download the Coco dataset image and annotation files using commands in the terminal
  - train images: `!wget http://images.cocodataset.org/zips/train2017.zip`
  - val images: `!wget http://images.cocodataset.org/zips/val2017.zip`
  - annotation: `!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
- Update the directory path wherever necessary in the code files

2. **Running Experiments**:
   The experiments were originally run on graphics Nvidia RTX3090 with 24 GB GPU memory, although it has been observed that on average maximum memory usage for creating the dataset should be up to 16 Gigabytes and training/evaluation should consume up to 8 Gigabytes of GPU memory.
- ***Prompt to Image Dataset***: Run the command `python promptImageDataset/*_filter_coco.py` to filter out the working set of prompts and images necessary for our training and `python promptImageDataset/*_dataset.py` to generate the dataset using the eligible images. Batch size and number of prompts can be set in individual scripts.
- ***Training***: Run the command `python mergeTextPrediction/mtp_training.py` to train the MLP model. The same script can be updated to train different types of models such as object, quality, empty, and combined. The model can be trained using dataset files created in the previous step or existing dataset files found at the share drive link (Ref: **)
- ***Evaluation***: Run the command `python mergeTextPrediction/mtp_evaluation.py` to evaluate the trained model. This repository also contains the pre-trained models trained on a large dataset which can be used directly to run experiments.

## Acknowledgments

This research was conducted under the guidance of Professor Jin Sun. Special thanks to the committee members, Professor Lakshmish Ramaswamy and Professor Wei Niu, for their support and encouragement. Gratitude is also extended to the University of Georgia for providing the resources and financial support necessary for this study.

## Contact

For any inquiries or further information, please contact Vaibhav Goyal at vaibhavgoyal17796@gmail.com.
