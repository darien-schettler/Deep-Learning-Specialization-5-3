import numpy as np
from sklearn.utils import class_weight

# ----------------------
# LOAD FULL TRAINING SET
# ----------------------

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

Y_unique = np.unique(Y, axis=0)
for i in range(0, Y_unique.shape[0]):
    positives = sum(Y_unique[i,:,:])
    print("\nNUMBER {}\n".format(i))
    print("------ {} ONES -----".format(int(positives[0])))
    print(Y_unique[i,:,:])
    print()
'''
Y_ravel = np.zeros((Y.shape[0]*Y.shape[1],Y.shape[2]))

for i in range(0, Y.shape[0]):
    Y_ravel[(i*Y.shape[1]):((i+1)*Y.shape[1]), :] = Y[i, :, :]
print(sum(Y_ravel == 0))
'''

# ----------------------

# ----------------------
# LOAD FULL DEV SET
# ----------------------

# Load preprocessed dev set examples (REAL NOT SYNTHESIZED AUDIO)
#X_dev = np.load("./XY_dev/X_dev.npy")
#Y_dev = np.load("./XY_dev/Y_dev.npy")
