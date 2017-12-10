dataset_tools = dofile('utils/pascal_voc.lua')
classLabels = dataset_tools.classLabels
numClasses = dataset_tools.numClasses

print('Wating...\nLoading Dataset in:  ' .. opts.PATHS.DATASET_CACHED)

dataset = torch.load(opts.PATHS.DATASET_CACHED)
print('Dataset load done.')

dofile('utils/parallel_batch_loader.lua')
dofile('utils/example_loader.lua')

