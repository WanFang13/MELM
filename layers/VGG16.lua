return function(modelPath)
	local vgg16 = torch.load(modelPath)

	local conv_layers = nn.Sequential()
	for i = 1, 30 do
		conv_layers:add(vgg16:get(i))
	end

	local fc_layers = nn.Sequential()
	for i = 33, 38 do
		fc_layers:add(vgg16:get(i))
	end

	local fc_6 = nn.Sequential()
	for i = 33, 35 do
		fc_6:add(vgg16:get(i))
	end

	local fc_7 = nn.Sequential()
	for i = 36, 38 do
		fc_7:add(vgg16:get(i))
	end

	return {
		conv_layers = conv_layers, 
		fc_layers = fc_layers, 
		fc_6 = fc_6, 
		fc_7 = fc_7, 
		channel_order = 'bgr', 
		spatial_scale = 1 / 16, 
		fc_layers_output_size = 4096,
		pooled_height = 7, 
		pooled_width = 7, 
		spp_correction_params = {offset0 = -18, offset = 0.0},
		--spp_correction_params = {offset0 = -18.0, offset = 9.5},
		fc_layers_view = function(RoiReshaper) return nn.View(-1):setNumInputDims(3) end,
		normalization_params = {channel_order = 'bgr', rgb_mean = {122.7717, 115.9465, 102.9801}, rgb_std = nil, scale = 255}
	}
end
