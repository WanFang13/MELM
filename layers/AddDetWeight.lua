local AddDetWeight, Parent = torch.class('nn.AddDetWeight', 'nn.Module')

local eps = 1e-12

function AddDetWeight:__init()
  Parent.__init(self)
  self.ObjectnessScore = nil
  self.AddWeightPattern = SETTINGS.Recurrent_Pattern
end

function AddDetWeight:updateOutput(input)
  -- input
  -- input[1] : nRoi * nDim (nDim=4096 for fc7)
  -- input[2] : nRoi * nCls+1 (detector)

  local nDim = input[1]:size(2)
  local nCls = input[2]:size(2)-1

  if not epoch_id then
    epoch_id = SETTINGS.test_epoch_num
  end

  local aneal_rate = nil

  if self.AddWeightPattern == 'None' then
    aneal_rate = 0
  end
  if self.AddWeightPattern == '0.5' then
    aneal_rate = 0.5
  end
  if self.AddWeightPattern == 'Anneal0.63' then
    aneal_rate = (epoch_id-1)/SETTINGS.NUM_EPOCHS/1.5
  end
  if self.AddWeightPattern == 'Anneal0.5' then
    aneal_rate = (epoch_id-1)/SETTINGS.NUM_EPOCHS/2
  end
  assert(aneal_rate, 'Recurrent_Pattern not supperted!')
  
  self.ObjectnessScore = input[#input][{{},{1,nCls}}]:max(2):repeatTensor(1,nDim)
  self.ObjectnessScore = (1-aneal_rate)+aneal_rate*self.ObjectnessScore

  self.output = {}
  self.output[1] = torch.cmul(input[1],self.ObjectnessScore)
  for i=1, #input-1 do
    self.output[i+1] = input[i+1]
  end

  return self.output
end

function AddDetWeight:updateGradInput(input, gradOutput)
  self.gradInput = {}
  self.gradInput[1] = torch.cmul(gradOutput[1],self.ObjectnessScore)

  for i=1, #gradOutput-1 do
    self.gradInput[i+1] = gradOutput[i+1]
  end
  return self.gradInput
end