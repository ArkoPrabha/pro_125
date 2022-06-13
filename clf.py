import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os,ssl,time
from PIL import Image
import PIL.ImageOps

x=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scaled=x_train/255
x_test_scaled=x_test/255

clf=LogisticRegression(solver='saga',multi_class='multinomial')
clf.fit(x_train_scaled,y_train)

y_pred=clf.predict(x_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
#print(accuracy)

def get_pred(image):
        im_pil=Image.open(image)
        image_bw=im_pil.convert('L')
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
        pixel_filter=20
        min_pixel=np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel=np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        
        test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred=clf.predict(test_sample)
        
        return test_pred[0] 