from flask import Flask
from flask_wtf.csrf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

app = Flask(__name__)

csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = '424f396696e4aa10887285e12953942a'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

app.config['UPLOAD_FOLDER'] = 'D:/Final Year Project/User Uploads'
#Maximum file size in bytes
app.config['MAX_FILE_SIZE'] = 10485760

from contents import routes