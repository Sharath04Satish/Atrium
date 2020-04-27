from keras.models import load_model
from keras.utils import to_categorical 
import pandas as pd
import glob
import os
import numpy as np
from collections import Counter

model = load_model('static/models/keras_model.h5')

files = glob.glob('D:\\Atrium\\User Uploads\\*') 
latest_file = max(files, key=os.path.getctime)

df = pd.read_csv(latest_file, header=None)
X_test_upload = df.values[:, :-1]
y_test_upload = df.values[:, -1].astype(int)

y_test_upload = to_categorical(y_test_upload)

prediction = model.predict(X_test_upload)

prediction = np.asarray(prediction)
max_values = np.argmax(prediction, axis=1)


    

