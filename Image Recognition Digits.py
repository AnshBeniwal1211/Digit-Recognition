#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install extra-keras-datasets')


# In[2]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from extra_keras_datasets import emnist


# In[3]:


(x_train, y_train), (x_test, y_test) = emnist.load_data(type='digits')


# In[4]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[5]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(63, activation='softmax'))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[6]:


model.fit(x_train, y_train,  epochs=3)
model.save('handwritten2.model')


# In[7]:


model = tf.keras.models.load_model('handwritten2.model')
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)


# In[11]:


image_number = 1
while os.path.isfile(f"C:\\Users\\Beniwal\\Downloads\\Desktop\\Neural Project\\Digit Recognition\\Digit{image_number}.png"):
    try:
        img = cv2.imread(f"C:\\Users\\Beniwal\\Downloads\\Desktop\\Neural Project\\Digit Recognition\\Digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Some Error Occured!!!!")
    finally:
        image_number += 1


# In[ ]:





# In[ ]:





# In[ ]:




