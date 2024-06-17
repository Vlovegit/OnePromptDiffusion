from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch

# Initialize the processor and model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def detect_object(image_path, prompt_list, negative_prompt, full_prompt, type):
    # Load your image
    image = Image.open(image_path)

    prompts = prompt_list

    if negative_prompt not in prompts:
        prompts.append(negative_prompt)

    # Process the image and prompts
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

    # Predict the segmentation
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply sigmoid to get probabilities and calculate the presence
    preds = torch.sigmoid(outputs.logits).squeeze(1)  # Squeeze to remove the batch dimension

    # Define a threshold for detecting the presence of the object
    if type is True:
        area_threshold = 0.1
    else:
        area_threshold = 0.15  # Percentage of the image area

    for i, prompt in enumerate(prompts):
        # Calculate the percentage of the image area that the mask covers
        mask_area = preds[i].mean().item()

        # Check if the mask area exceeds the threshold
        object_present = mask_area > area_threshold

        print(f"Object '{prompt}' present: {object_present} (Area: {mask_area:.2%})")

        if negative_prompt in prompt:
            return object_present, mask_area, area_threshold

    return object_present

