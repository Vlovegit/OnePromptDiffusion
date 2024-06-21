device='cuda:0'
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = '7.5'

# Dataset file configuration
object_dataset = '/home/myid/vg80700/gits/dataset_object'
quality_dataset = '/home/myid/vg80700/gits/dataset_quality'
empty_dataset = '/home/myid/vg80700/gits/dataset_empty'
file_batchsize = 5

# Prominent object cocoids directory
object_cocoids = '/home/myid/vg80700/gits/results_object'
quality_cocoids = '/home/myid/vg80700/gits/results_quality'
empty_cocoids = '/home/myid/vg80700/gits/results_empty'

#Pretrained model configuration
object_model = '/home/myid/vg80700/gits/models_object'
quality_model = '/home/myid/vg80700/gits/models_quality'
empty_model = '/home/myid/vg80700/gits/models_empty'

# FID Metric temporary directory
fid_metric = '/home/myid/vg80700/gits/fid_metric'

#Performance analysis file path
performance_analysis = '/home/myid/vg80700/gits/performance_analysis'

#Coco dataset Configuration
image_directory = '/home/myid/vg80700/gits/coco_train2017/train2017'
annotation_file = '/home/myid/vg80700/gits/coco_ann2017/annotations/captions_train2017.json'
annotation_keyfile = '/home/myid/vg80700/gits/coco_ann2017/annotations/instances_train2017.json'

#Directory Paths
utils =  '/home/myid/vg80700/gits/OnePromptDiffusion/utils'
mto = '/home/myid/vg80700/gits/OnePromptDiffusion/mergeTextOptimization'