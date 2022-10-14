'''
The file utilized Principal Component Analysis to find the primary features of two subjects from the YaleFace dataset and use that to be able to detect the subject in other images.
'''

import numpy as np
from numpy import asarray
import math
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as spio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
from sklearn.decomposition import PCA
from PIL import Image
import pandas as pd
from skimage.measure import block_reduce
from skimage.transform import resize

#create data matrix for subject 1 and 2
im = []
images1 = ['glasses.gif','happy.gif','leftlight.gif','noglasses.gif','normal.gif','rightlight.gif','sad.gif','sleepy.gif','surprised.gif','wink.gif']
for i in images1:
    x1 = plt.imread('data/yalefaces/subject01.' + i)
    x1 = resize(x1,(x1.shape[0]//4,x1.shape[1]//4),anti_aliasing=True)
    im.append(x1)
y1 = np.array(im)
m1,h,w = y1.shape
sub1_v = np.reshape(y1,(m1,h*w),order='C')

im2 = []
images2 = ['glasses.gif','happy.gif','leftlight.gif','noglasses.gif','normal.gif','rightlight.gif','sad.gif','sleepy.gif','wink.gif']
for i in images2:
    x2 = plt.imread('data/yalefaces/subject02.' + i)
    x2 = resize(x2,(x2.shape[0]//4,x2.shape[1]//4),anti_aliasing=True)
    im2.append(x2)
y2 = np.array(im2)
m2,h,w = y2.shape
sub2_v = np.reshape(y2,(m2,h*w),order='C')

#%%
k = 6
#subject 1
#find mean
mu_sub1 = np.mean(sub1_v,axis=0)
xmu_sub1 = sub1_v - mu_sub1
#create covariance matrix
c_sub1 = np.cov(xmu_sub1.T)
#SVD
vecs, vals, _ = ll.svds(c_sub1,k=k)
ind = np.argsort(vals)
vals= vals[ind]
vecs = vecs[:,ind].T

#%%
#subject 2
#find mean
mu_sub2 = np.mean(sub2_v,axis=0)
xmu_sub2 = sub2_v - mu_sub2
#create covariance matrix
c_sub2 = np.cov(xmu_sub2.T)
#SVD
vecs2, vals2, _ = ll.svds(c_sub2,k=k)
ind2 = np.argsort(vals2)
vals2= vals2[ind2]
vecs2 = vecs2[:,ind2].T

#subject 1 eigenfaces
print('Eigenfaces for subject 1')
fig = plt.figure(figsize=(8, 6))
for i in range(6):
    ax = fig.add_subplot(5, 3, i + 1, xticks=[], yticks=[])
    ax.imshow(np.reshape(vecs[i],(60,80)), cmap=plt.cm.bone)
plt.show()

#subject 2 eigenfaces
print('Eigenfaces for subject 2')
fig = plt.figure(figsize=(8, 6))
for i in range(6):
    ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(np.reshape(vecs2[i],(60,80)), cmap=plt.cm.bone)
plt.show()

#downsample test subject pics
test1 = resize(plt.imread('data/yalefaces/subject01-test.gif'),(h*w,1),anti_aliasing=True)
test2 = resize(plt.imread('data/yalefaces/subject02-test.gif'),(h*w,1),anti_aliasing=True)

#find residuals
print('Subject 1 Face Detection:')
s11 = np.linalg.norm((test1-mu_sub1) - vecs[0] * np.dot(vecs[0].T,(test1-mu_sub1)))
s12 = np.linalg.norm((test1-mu_sub1) - vecs2[0] * np.dot(vecs2[0].T,(test1-mu_sub1)))
print('s1_1:',s11)
print('s1_2:',s12)

print('Subject 2 Face Detection')
s21 = np.linalg.norm((test2-mu_sub2) - vecs[0] * np.dot(vecs[0].T,(test2-mu_sub2)))
s22 = np.linalg.norm((test2-mu_sub2) - vecs2[0] * np.dot(vecs2[0].T,(test2-mu_sub2)))
print('s2_1:',s21)
print('s2_2:',s22)


