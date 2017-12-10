model = nn.Sequential():
  add(nn.ParallelTable():
    add(base_model.conv_layers):
    add(nn.Identity())
  ):
  add(RectangularRingRoiPooling(base_model.pooled_height, base_model.pooled_width, base_model.spatial_scale, base_model.spp_correction_params)):
  add(RoiReshaper:StoreShape()):
  add(base_model.fc_layers_view(RoiReshaper)):
  add(base_model.fc_layers):
  add(nn.ConcatTable():
    add(nn.Identity()):
    add(nn.Sequential():
      add(nn.Linear(base_model.fc_layers_output_size, numClasses+1):named('det_fc81')):
      add(cudnn.SpatialSoftMax())
    ):
    add(nn.Sequential():
      add(nn.Linear(base_model.fc_layers_output_size, numClasses+1):named('det_fc82')):
      add(cudnn.SpatialSoftMax())
    )
  ):
  add(nn.AddDetWeight()):
  add(nn.ParallelTable():
    add(nn.Sequential():
      add(nn.ConcatTable():
        add(nn.Sequential():
          add(nn.Linear(base_model.fc_layers_output_size, numClasses):named('sfc8c')):
          add(nn.Squeeze(1))
        ):
        add(nn.Sequential():
          add(nn.Linear(base_model.fc_layers_output_size, numClasses):named('sfc8d')):
          add(RoiReshaper:RestoreShape(4))
        )
      ):
      add(nn.MinEntropy())
    ):
    add(nn.Identity()):
    add(nn.Identity())
  ):
  add(nn.ADEntropy())

criterion_image = nn.BCECriterion()
criterion_box   = nn.ClassNLLCriterion()
criterion_box2   = nn.ClassNLLCriterion()

criterion = nn.ParallelCriterion():
  add(criterion_image, 1):
  add(criterion_box, SETTINGS.DetRate1):
  add(criterion_box2, SETTINGS.DetRate2)

optimState          = {
  learningRate = SETTINGS.LearningRate, 
  momentum     = 0.9, 
  weightDecay  = 5e-4
}
optimState_annealed = {
  learningRate = SETTINGS.LearningRateAneal, 
  momentum     = 0.9, 
  weightDecay  = 5e-4, 
  epoch        = SETTINGS.AnnealEpoch
}
