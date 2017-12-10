require 'orn'
require 'cuorn'

torch.setdefaulttensortype('torch.FloatTensor')
precision = 5e-4

-- ORAlign
cpu = nn.ORAlign(4)
gpu = cpu:clone():cuda()
input = torch.rand(2, 4, 1, 1)
output_cpu = cpu:forward(input)
output_gpu = gpu:forward(input:cuda())
err = math.abs(torch.sum(output_gpu:float() - output_cpu))
if err < precision then
    print("[ORAlign] outputs OK", err)
else
    print("[ORAlign] outputs are not equal", err)
end
gradInput_cpu = cpu:backward(input, output_cpu)
gradInput_gpu = gpu:backward(input:cuda(), output_gpu)
err = math.abs(torch.sum(gradInput_gpu:float() - gradInput_cpu))
if err < precision then
    print("[ORAlign] gradInputs OK", err)
else
    print("[ORAlign] gradInputs are not equal", err)
end

-- ORConv
cpu = nn.ORConv(2,2,8,3,3,1,1)
gpu = cpu:clone():cuda()
input = torch.rand(5, 2*8, 7, 7)
output_cpu = cpu:forward(input)
output_gpu = gpu:forward(input:cuda())
err = math.abs(torch.sum(output_gpu:float() - output_cpu))
if err < precision then
    print("[ORConv] outputs OK", err)
else
    print("[ORConv] outputs are not equal", err)
end
gradInput_cpu = cpu:backward(input, output_cpu)
gradInput_gpu = gpu:backward(input:cuda(), output_gpu)
err = math.abs(torch.sum(gradInput_gpu:float() - gradInput_cpu))
if err < precision then
    print("[ORConv] gradInputs OK", err)
else
    print("[ORConv] gradInputs are not equal", err)
end
cpu:updateParameters(0.1)
gpu:updateParameters(0.1)
err = math.abs(torch.sum(gpu.weight:float() - cpu.weight))
if err < precision then
    print("[ORConv] weights OK", err)
else
    print("[ORConv] weights are not equal", err)
end
err = math.abs(torch.sum(gpu.bias:float() - cpu.bias))
if err < precision then
    print("[ORConv] biases OK", err)
else
    print("[ORConv] biases are not equal", err)
end


print("all done")
