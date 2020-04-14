from flask import render_template, url_for, redirect
from contents import app, db, bcrypt
from contents.forms import SignInForm, SignUpForm
from contents.models import Users
from flask_login import login_user
from contents.file_upload import CSVFile
from werkzeug.utils import secure_filename
import os
from glob import glob
from keras.models import load_model
from keras.utils import to_categorical 
import numpy as np
import pandas as pd
from collections import Counter

model = load_model('D:\\Atrium\\atrium_model.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signin', methods=('GET', 'POST'))
def signIn():
    form = SignInForm()
    if form.validate_on_submit():
        user = Users.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('index'))
    return render_template('signin.html', form=form)


@app.route('/signup', methods=('GET', 'POST'))
def signUp():
    form = SignUpForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('UTF-8')
        user = Users(name=form.name.data, email=form.email.data, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('signIn'))

    return render_template('signup.html', form=form)


@app.route('/upload', methods=('GET', 'POST'))
def csvUpload():
    form = CSVFile()

    if form.validate_on_submit():
        f = form.file.data
        filename = secure_filename(f.filename)
        os.chdir('D:\\Atrium\\User Uploads')
        f.save(os.path.abspath(filename))

        files = glob('D:\\Atrium\\User Uploads\\*') 
        latest_file = max(files, key=os.path.getctime)
        print(latest_file)

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
        print(output)

        for file in files:
            os.remove(file)

        return render_template('upload.html', form=form, output=output, showReport=True)

    return render_template('upload.html', form=form, showReport=False)


@app.route('/resuts')
def results():
    return render_template('results.html', output=output)