local FixProb, Parent = torch.class('nn.FixProb', 'nn.Module')

local eps = 1e-4

function FixProb:__init()
	Parent.__init(self)
end

function FixProb:updateOutput(input)
	for ind = 1, 20 do
		if input[{1,ind}] > 1-eps and batch_labels_gpu[{1,ind}] == 0 then
			print("val= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			print(input[{1,ind}])
			input[{1,ind}] =1 - eps
		end
		if input[{1,ind}] < eps   and batch_labels_gpu[{1,ind}] == 1 then
			print("val= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			print(input[{1,ind}])			
			input[{1,ind}] = eps
		end
	end
	
	self.output=input:clone()
	return self.output
end

function FixProb:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:clone()
	return self.gradInput
end