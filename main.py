import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 

from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43 

for i in range(0,43): 
    path = "data/Train/" + str(i)
    images = os.listdir(path) 
    for a in images: 
        try: 
            image = Image.open(path + '/'+ a) 
            image = image.resize((30,30)) 
            image = np.array(image) 
            data.append(image) 
            labels.append(i) 
        except: 
            print("Error loading image")
            break
    print(f"file {i}")
    
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)

y_t1 = to_categorical(y_t1, 43)
y_t2 = to_categorical(y_t2, 43)