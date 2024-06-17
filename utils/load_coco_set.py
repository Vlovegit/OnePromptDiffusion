import torchvision.transforms as tra
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
import json
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import torch
import os

#?set file paths
image_directory = '/home/myid/vg80700/gits/train2017'
annotation_file = '/home/myid/vg80700/gits/annotations/captions_train2017.json'
annotation_keyfile = '/home/myid/vg80700/gits/annotations/instances_train2017.json'

# Initialize COCO API
coco = COCO(annotation_keyfile)

#?load the coco captions
def prepare_coco_dataset():
    # Load annotation file
       
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Create a dictionary to store annotations
    annotations_dict = {}

    # Populate the dictionary with image IDs and corresponding annotations
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']

        if image_id in annotations_dict:
            annotations_dict[image_id].append(caption)
        else:
            annotations_dict[image_id] = [caption]

    return annotations_dict


def get_image(image_id:str)->torch.Tensor:

    #?setting up some image transforms to cleanup the original COCO image (not strictly neccesary)
    #?this is what we use for inpainting
    image_transform_p = transforms.ToTensor()
    image_transform_reshape = transforms.Resize((512,512))
    image_transform_b = lambda x: image_transform_p(x) * 2 -1
    image_transform = lambda x: image_transform_reshape(image_transform_b(x))
    mask_reshape = transforms.Resize((512,512), interpolation=Image.NEAREST)

    seg_list = []
    image_info = coco.loadImgs(image_id)[0]
    try:
        image_path = os.path.join(image_directory, image_info['file_name'])
    except Exception as e: print(e)
    image = Image.open(image_path)

    # Load the annotations for the selected image
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    for ann in annotations:
        if 'segmentation' in ann:
            seg_list.append((mask_reshape( torch.from_numpy(coco.annToMask(ann)).unsqueeze(0) ), ann['category_id']))


    return image_transform(image), seg_list, image


# Loading the coco dataset prompts
def load_coco_dataset(no_of_prompts:int, skip_count:int=0):

    
    #?get the annotation keys and name pairs
    with open( annotation_keyfile, 'r') as f:
        annotation_categories = json.load(f)['categories']
        annotation_categories = {x['id']: x['name'] for x in annotation_categories}

    #Generate Dataset reference annotations
    annotations_dict = prepare_coco_dataset()

    #?setting the number of examples used
    length_of_testing_set = no_of_prompts + skip_count

    #?setting the list of prompt keys
    testingKeys = list(annotations_dict.keys())[skip_count:length_of_testing_set]

    #?set the number of pixels that the mask must be larger than for it to be considered as a example
    mask_bound = 75


    #?the count of prompts which did not have any categories from COCO and therefore we pass over
    vbn_c = 0

    prompt = []
    negative_prompt = []
    coco_id = []
    images = []

    for key in testingKeys:

        # print('I am here')

        # print('Prompt : ', key)

        #?generate prompts and np
        caption = annotations_dict[key][0]

        # print('Caption : ', caption)

        #?load the oriignal COCO image and masks (to help pick the negative prompt)
        try:
            image, masks, src_image = get_image(key)
            mask_ord = sorted(masks, key=lambda x: torch.sum(x[0]), reverse=True)[0]
            mask = mask_ord[0]
            run_exec = True

            assert torch.sum(mask) > mask_bound, "skipping for to small mask"

        except Exception as e:
            print(f"Exception: {e}")
            run_exec = False
            vbn_c += 1

        #?skip this prompt if there are no useful masks
        if not run_exec: 
            # print('Skipping this prompt')
            continue

        #get the name of the segmentation
        segmentation_word = annotation_categories[mask_ord[1]]
        #?generate the prompt & n prompt
        # caption = caption.replace(segmentation_word, '')
        prompt.append(caption)
        negative_prompt.append(segmentation_word)
        coco_id.append(key)
        images.append(image)

    return coco_id, prompt, negative_prompt, images


def load_coco_ids(coco_ids:list):
    # get the annotation keys and name pairs
    with open(annotation_keyfile, 'r') as f:
        annotation_categories = json.load(f)['categories']
        annotation_categories = {x['id']: x['name'] for x in annotation_categories}

    # Generate Dataset reference annotations
    annotations_dict = prepare_coco_dataset()  # Assuming prepare_coco_dataset is defined elsewhere

    # set the number of pixels that the mask must be larger than for it to be considered as an example
    mask_bound = 75

    # the count of prompts which did not have any categories from COCO and therefore we pass over
    vbn_c = 0
    coco_id_list = []
    prompt_list = []
    negative_prompt_list = []
    image_list = []

    # find the entry corresponding to the given coco_id
    for coco_id in coco_ids:

        key = int(coco_id)

        if key not in annotations_dict:
            continue

        # generate prompts and np
        prompt = annotations_dict[key][0]

        # load the original COCO image and masks (to help pick the negative prompt)
        try:
            image, masks, src_image = get_image(key)  # Assuming get_image is defined elsewhere
            mask_ord = sorted(masks, key=lambda x: torch.sum(x[0]), reverse=True)[0]
            mask = mask_ord[0]
            run_exec = True

            assert torch.sum(mask) > mask_bound, "skipping for too small mask"

        except Exception as e:
            print(f"Exception: {e}")
            run_exec = False
            vbn_c += 1

        # skip this prompt if there are no useful masks
        if not run_exec:
            raise ValueError(f"No valid mask for COCO ID {coco_id}")

        # get the name of the segmentation
        negative_prompt = annotation_categories[mask_ord[1]]
        prompt_list.append(prompt)
        negative_prompt_list.append(negative_prompt)
        image_list.append(image)
        coco_id_list.append(key)


    return coco_id_list, prompt_list, negative_prompt_list, image_list