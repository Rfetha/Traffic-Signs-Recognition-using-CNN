from main import X_t1, X_t2, y_t1, y_t2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, ReLU, InputLayer
from sklearn.metrics import accuracy_score


model = Sequential()
model.add(InputLayer(input_shape=X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epoch = 50
anc = model.fit(X_t1, y_t1, batch_size=32, epochs=epoch, validation_data=(X_t2, y_t2))        #anc = history
model.save("my_model.h5")



"""

plt.figure(0)
plt.plot(anc['accuracy'], label='training accuracy')
plt.plot(anc['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(anc['loss'], label='training loss')
plt.plot(anc['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()"""


y_test = pd.read_csv('data\Test.csv')
labels = y_test["ClassId"].values
imgs =  "data/" + y_test["Path"].values
#print(imgs)


data=[]
for img in imgs:
   image = Image.open(img)
   image = image.resize((30,30))
   data.append(np.array(image))
X_test=np.array(data)

predict_x=model.predict(X_test)
classes_x=np.argmax(predict_x, axis=1)
print(predict_x, classes_x)

from sklearn.metrics import accuracy_score
print(accuracy_score(labels, classes_x))


model.save("traffic_classifier.h5")