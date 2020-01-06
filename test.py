# -*- coding: utf-8 -*-
"""
@author: jinning
"""

import os
import numpy as np
from mylib.densenet import get_compiled
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 变量初始化
batch_size = 32
ckg = 32
h_ckg = int(ckg / 2)

x_test = np.ones((117, ckg, ckg, ckg))
# 读取测试集
i = 0
path = "dataset/test"  
path_list = os.listdir(path)
path_list.sort() 

for filename in path_list:
    tmp = np.load(os.path.join(path, filename))

    voxel = tmp['voxel']
    seg = tmp['seg']

    x_test[i] = (voxel * seg)[50 - h_ckg:50 + h_ckg, 50 - h_ckg:50 + h_ckg, 50 - h_ckg:50 + h_ckg]
    i = i + 1

x_test = x_test.reshape(x_test.shape[0], ckg, ckg, ckg, 1)

# compile model
model = get_compiled()
model_path1 = 'weights/weight.h5'
model.load_weights(model_path1)
test1 = model.predict(x_test, batch_size, verbose=1)


col0 = np.loadtxt("sampleSubmission.csv", str, delimiter=",", skiprows=1, usecols=0)
path = "SubmissionFile.csv"
np.savetxt(path, np.column_stack((col0,test1[:, 1])), delimiter=',', fmt='%s', header='Id,Predicted', comments='')
