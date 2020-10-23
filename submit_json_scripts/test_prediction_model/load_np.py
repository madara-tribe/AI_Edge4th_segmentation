import numpy as np
pred = np.load('test_prediction.npy')
print(pred.shape)
print(np.unique(pred))
