-- settings for path and models
dofile('utils/settings.lua')
dofile('utils/preprocess.lua')

dofile('utils/opts.lua')
dofile('utils/util.lua')
dofile('utils/dataset.lua')
threads = require 'threads'
print('Starting...')
local MATLAB = assert((#sys.execute('which matlab') > 0 and 'matlab -nodisplay -r') or (#sys.execute('which octave') > 0 and 'octave --eval'), 'matlab or octave not found in PATH')
local subset = SETTINGS.SUBSET_FOR_TESTING

print(arg[2+9])
print(arg)

local check_input = false
if arg[2+9] == '1' then
	opts.NMS_OVERLAP_THRESHOLD = 0.4
	opts.NMS_SCORE_THRESHOLD = 1e-4
	check_input = true
	output_field = opts.OUTPUT_FIELDS[1]
	prefix = '-global'
	print('detect_mAP: ' .. opts.OUTPUT_FIELDS[1])
end
if arg[2+9] == '2' then
	opts.NMS_OVERLAP_THRESHOLD = 0.3
	opts.NMS_SCORE_THRESHOLD = 5e-3
	check_input = true
	output_field = opts.OUTPUT_FIELDS[2]
	prefix = '-local1'
	print('detect_mAP: ' .. opts.OUTPUT_FIELDS[2])
end
if arg[2+9] == '3' then
	opts.NMS_OVERLAP_THRESHOLD = 0.3
	opts.NMS_SCORE_THRESHOLD = 5e-3
	check_input = true
	output_field = opts.OUTPUT_FIELDS[3]
	prefix = '-local2'
	print('detect_mAP: ' .. opts.OUTPUT_FIELDS[3])
end
if arg[2+9] == '2+3' then
	opts.NMS_OVERLAP_THRESHOLD = 0.3
	opts.NMS_SCORE_THRESHOLD = 5e-3
	check_input = true
	evalue_both = true
	output_field = 'output_locals'
	prefix = '-locals'
	print('detect_mAP: locals')
end

if not arg[2+9] or not check_input then
	print('Please choose the right output field (1 2 3 or 2+3)!')
	os.exit()
end


opts.SCORES_FILES = arg
rois = hdf5_load(opts.SCORES_FILES[1+9], 'rois')
scores = {}
if evalue_both then
	scores_1 = hdf5_load(opts.SCORES_FILES[1+9], 'outputs/' .. opts.OUTPUT_FIELDS[2])
	scores_2 = hdf5_load(opts.SCORES_FILES[1+9], 'outputs/' .. opts.OUTPUT_FIELDS[3])
	for exampleIdx = 1, #scores_1 do
		scores[exampleIdx] = scores_1[exampleIdx]:clone()
		scores[exampleIdx] = scores[exampleIdx]:add(scores_2[exampleIdx])/2
	end
else
	local i=1
	scores_i = hdf5_load(opts.SCORES_FILES[i+9], 'outputs/' .. output_field)
	
	for exampleIdx = 1, #scores_i do
		if not scores_i[exampleIdx] then
			scores[exampleIdx] = scores_i[tostring(exampleIdx)]:clone()
			assert(scores[exampleIdx])
		else
			scores[exampleIdx] = scores_i[exampleIdx]:clone()
		end
	end
end

local detrespath = dataset_tools.package_submission(
	opts.PATHS.VOC_DEVKIT_VOCYEAR, 
	dataset,
	opts.DATASET, 
	subset, 
	'comp4_det', 
	rois, 
	scores, 
	nms_mask(rois, scores, opts.NMS_OVERLAP_THRESHOLD, opts.NMS_SCORE_THRESHOLD)
)
local opts = opts

print('data process done')

if dataset[subset].objectBoxes == nil then
	print('detection mAP cannot be computed for ' .. opts.DATASET .. '. Quitting.')
	print(('VOC submission saved in "%s/results-%s-%s%s.tar.gz"'):format(SETTINGS.RESULT_SAVE_FOLDER, 'comp4_det', subset, prefix))
	os.exit(0)
end

res = {[output_field] = {_mean = nil, by_class = {}}}
APs = torch.FloatTensor(numClasses):zero()

local imgsetpath = paths.tmpname()
os.execute(('sed \'s/$/ -1/\' %s > %s'):format(paths.concat(opts.PATHS.VOC_DEVKIT_VOCYEAR, 'ImageSets', 'Main', subset .. '.txt'), imgsetpath)) -- hack for octave

print('evaluating')
jobQueue = threads.Threads(math.min(32,numClasses)) --numClasses
for classLabelInd, classLabel in ipairs(classLabels) do
	jobQueue:addjob(function()
		os.execute(('%s "oldpwd = pwd; cd(\'%s\'); addpath(fullfile(pwd, \'VOCcode\')); Dataset = \'%s\'; testset = \'%s\'; VOCinit; cd(oldpwd); VOCopts.testset = \'%s\'; VOCopts.detrespath = \'%s\'; VOCopts.imgsetpath = \'%s\'; classLabel = \'%s\'; [rec, prec, ap] = VOCevaldet(VOCopts, \'comp4\', classLabel, false); dlmwrite(sprintf(VOCopts.detrespath, \'resu4\', classLabel), ap); quit;"'):format(MATLAB, paths.dirname(opts.PATHS.VOC_DEVKIT_VOCYEAR), opts.DATASET, subset, subset, detrespath, imgsetpath, classLabel))
		return tonumber(io.open(detrespath:format('resu4', classLabel)):read('*all'))
	end, function(ap) res[output_field].by_class[classLabel] = ap; APs[classLabelInd] = ap; end)
end
jobQueue:synchronize()
os.execute('[ -t 1 ] && reset')

res[output_field]._mean = APs:mean()

json_save(opts.PATHS.DETECTION_MAP:format(SETTINGS.test_epoch_num, prefix), res)
print('result in ' .. opts.PATHS.DETECTION_MAP:format(SETTINGS.test_epoch_num, prefix))
