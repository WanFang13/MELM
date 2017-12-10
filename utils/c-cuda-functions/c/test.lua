require 'salc'

-- jacobian test
-- ORAlign
jac = nn.Jacobian

size = math.random(3, 5)

input = torch.rand(2, size* 4, 1, 1)
module = nn.ORAlign(4)

err = jac.testJacobian(module, input)
print(string.format('[ORAlign] state error: %e', err))


-- ORConv
size = math.random(3, 5)

input = torch.rand(4, size, size)
module = nn.ORConv(1, 2, 4, 1, 1, 1, 1)

err = jac.testJacobian(module, input)
print(string.format('[ORConv] state error: %e', err))

err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
print(string.format('[ORConv] weight  error: %e', err))

err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
print(string.format('[ORConv] bias error: %e', err))

err = jac.testJacobianUpdateParameters(module, input, module.weight)
print(string.format('[ORConv] update weight error: %e', err))

err = jac.testJacobianUpdateParameters(module, input, module.bias)
print(string.format('[ORConv] update weight error: %e', err))

for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
    print(string.format('[ORConv] %e error on weight [%s]', err, t))
end

