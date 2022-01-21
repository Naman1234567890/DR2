import cv2 
import numpy as np 
import pandas as pd
from tenacity import retry_unless_exception_type 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from PIL import Image 
import PIL.ImageOps 
import os, ssl, time
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context
x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["0","1","2","3","4","5","6","7","8","9"]
nclasses=len(classes)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=8,train_size=7500,test_size=2500)
xtrainscaled=xtrain/255.0
xtestscaled=xtrain/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xtrainscaled,ytrain)
ypredict=clf.predict(xtestscaled)
accuracy=accuracy_score(ytest,ypredict)
print(accuracy)