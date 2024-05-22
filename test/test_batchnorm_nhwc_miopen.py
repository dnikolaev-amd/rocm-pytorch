# Before running, set the below flags and then run
# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export MIOPEN_LOG_LEVEL=7
# python batchnorm-miopen.py 

import torch

m = torch.nn.BatchNorm2d(100).to("cuda:0")

# random NCHW tensor on GPU
input = torch.randn(20, 100, 5, 4).to("cuda:0")

# change tensor dims to NHWC
input = input.to(memory_format=torch.channels_last) # <-- comment/uncomment this line

# Observations:
#
# 1) If you comment line 15, memory format is NCHW and MIOpen is invoked. 
# But CK kernel is not selected/called. See portion of the log that gets generated.
# MIOpen(HIP): Info2 [SearchForSolutions] BnCKFwdTraining: Not applicable
# MIOpen(HIP): Info2 [SearchForSolutions] BnFwdTrainingSpatialSingle: Success.
# MIOpen(HIP): Info2 [PrepareInvoker] Preparing kernel: MIOpenBatchNormFwdTrainSpatial
#
#
# 2) If you leave line 15 uncommented, then memory format is NHWC. 
# The expectation is that MIOpen will be invoked and select the CK kernel. 
# However, I notice that MIOpen itself seems not to be invoked at all. 
# < NO MIOPEN LOG GETS GENERATED >


# call batch norm
output = m(input)

print(output.shape)
print(output.stride())
