import numpy as np


if __name__ == "__main__":
    a = np.random.rand(4, 4)
    print(a)
    randMat = np.mat(np.random.rand(4, 4))
    print(randMat)
    print(randMat.I)
    invRandMat = randMat.I
    print(invRandMat * randMat)
    myEye = invRandMat * randMat
    e = myEye - np.eye(4)
    print(e)