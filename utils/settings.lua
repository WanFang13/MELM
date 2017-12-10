require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

--setting MODELs
require 'cudnn'
require 'salc'
require 'cusalc'

dofile 'layers/MinEntropy_Fast.lua'
dofile 'layers/ADEntropyRecurrent_acc.lua'
dofile 'layers/AddDetWeight.lua'
dofile 'layers/FixProb.lua'

--===========================================

--setting gpu id
local device_id = tonumber(arg[1])
cutorch.setDevice(device_id+1)

--setting for train and test
SETTINGS = {
	
	--common
	DATASET             = arg[2],  -- VOC2007, VOC2012
	PROPOSALS           = arg[4],  -- SSW, EB
	BASE_MODEL          = arg[3],  --'VGG16'
	model_path          = 'layers/MELM_model.lua',
	
	--training
	NUM_EPOCHS          = 20,
	LearningRate        = 5e-3,
	LearningRateAneal   = 5e-4,
	AnnealEpoch         = 15,
  DetRate1            = tonumber(arg[5]),
  DetRate2            = tonumber(arg[6]),
	Recurrent_Pattern   = arg[7],
	SUBSET              = 'trainval',

	--testing
	SUBSET_FOR_TESTING  = 'test',
	test_epoch_num      = tonumber(arg[8])
}

SETTINGS.RESULT_SAVE_FOLDER = 'ARL-' .. 
  SETTINGS.DetRate1 					.. '-' .. 
  SETTINGS.DetRate2 					.. '-' .. 
  SETTINGS.Recurrent_Pattern 	.. '-' .. 
  SETTINGS.PROPOSALS 					.. '-' ..
  arg[9]

SETTINGS.RESULT_SAVE_FOLDER = SETTINGS.DATASET .. 
	'/' .. SETTINGS.BASE_MODEL ..
	'/' .. SETTINGS.RESULT_SAVE_FOLDER

print('SETTINGS:')
print(SETTINGS)


