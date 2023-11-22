import secrets
from flask import Flask
from flask_bcrypt import Bcrypt

app = Flask(__name__)
bcrypt = Bcrypt(app)

pw = secrets.token_hex(16)
app.config['SECRET_KEY'] = pw

from app.routes import *