device='cuda:1'
NUM_DDIM_STEPS = '50'
GUIDANCE_SCALE = '7.5'


# Dataset file configuration
object_dataset = '/home/myid/vg80700/gits/dataset_object'
quality_dataset = '/home/myid/vg80700/gits/dataset_quality'
empty_dataset = '/home/myid/vg80700/gits/dataset_null'
file_batchsize = 200


# Prominent object cocoids directory
object_cocoids = '/home/myid/vg80700/gits/results_object'
quality_cocoids = '/home/myid/vg80700/gits/results_quality'
empty_cocoids = '/home/myid/vg80700/gits/results_null'


#Pretrained model configuration
object_model = '/home/myid/vg80700/gits/models_object'
quality_model = '/home/myid/vg80700/gits/models_quality'
empty_model = '/home/myid/vg80700/gits/models_null'


# FID Metric temporary directory
fid_metric = '/home/myid/vg80700/gits/fid_metric'

#Performance analysis file path
performance_analysis = '/home/myid/vg80700/gits/performance_analysis'