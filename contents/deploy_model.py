from keras.models import load_model
from keras.utils import to_categorical 
import numpy as np
import pandas as pd
from collections import Counter
from glob import glob
import os

def deployModel():
    files = glob('D:\\Atrium\\User Uploads\\*') 
    latest_file = max(files, key=os.path.getctime)

    model = load_model('D:\\Atrium\\atrium_model.h5')

    df = pd.read_csv(latest_file, header=None)
    X_test_upload = df.values[:, :-1]
    y_test_upload = df.values[:, -1].astype(int)

    y_test_upload = to_categorical(y_test_upload)

    prediction = model.predict(X_test_upload)

    prediction = np.asarray(prediction)
    max_values = np.argmax(prediction, axis=1)

    common = Counter(max_values).most_common()[0][0]
    arrhythmia_classes = {0: 'Normal', 1: 'Left Bundle Block', 2:'Right Bundle Block', 3: 'Premature Ventricular Contractions', 4: 'Paced Heartbeat'}
    output = arrhythmia_classes[common]
    return output




