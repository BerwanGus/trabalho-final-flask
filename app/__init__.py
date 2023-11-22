import secrets
from flask import Flask
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

app = Flask(__name__)
bcrypt = Bcrypt(app)

pw = secrets.token_hex(16)
app.config['SECRET_KEY'] = pw

engine = create_engine('sqlite:///instance/database.db', echo=True)

session = scoped_session(sessionmaker(bind=engine,
                                      autoflush=False,
                                      autocommit=False))

from app.routes import *