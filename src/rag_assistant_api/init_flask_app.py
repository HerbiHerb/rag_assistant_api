import os

from flask import Flask
from .local_database.database_models import db, User

import pathlib

DATABASE_ROOT = pathlib.Path(__file__).parent

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{DATABASE_ROOT}/instance/database.db"
)
app.secret_key = "140918"
db.init_app(app)


# with app.app_context():
#     db.create_all()

# with app.app_context():
#     user = User(username="Testuser", password="test")
#     db.session.add(user)
#     db.session.commit()
