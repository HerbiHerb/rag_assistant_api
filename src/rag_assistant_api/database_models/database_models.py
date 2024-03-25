import json
import os
from datetime import datetime
from typing import List

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import LONGTEXT

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    conversations = db.relationship("Conversation", backref="User")

    def check_username_and_password(username: str, password: str):
        user = User.query.filter(User.username == username).first()
        if user and user.username == username and user.password == password:
            return user.id
        else:
            return

    def save_new_user(username: str, password: str):
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return new_user.id

    def update_chat_messages(user_id: int, chat_messages: List):
        user = User.query.get(user_id)
        user.chat_messages = json.dumps(chat_messages)
        db.session.commit()

    def get_chat_messages(user_id: int):
        user = User.query.get(user_id)
        chat_messages = user.chat_messages
        if chat_messages != None:
            return json.loads(user.chat_messages)
        else:
            return []

    def check_user_exists(username: str):
        user = User.query.filter(User.username == username).first()
        if user != None:
            return True
        else:
            return False

    def __repr__(self):
        return f"<User {self.firstname}>"


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    chat_messages = db.Column(db.Text)
    timestamp_last_message = db.Column(db.DateTime, nullable=False)

    def generate_new_conversation(user_id: int):
        try:
            conversation = Conversation(user_id=user_id, timestamp_last_message=datetime.now())
            db.session.add(conversation)
            db.session.commit()
            return conversation.id
        except Exception as e:
            print(e)

    def update_chat_messages(conv_id: int, chat_messages: List):
        try:
            conversation = Conversation.query.get(conv_id)
            conversation.chat_messages = json.dumps(chat_messages)
            conversation.timestamp_last_message = datetime.now()
            db.session.commit()
        except Exception as e:
            print(e)

    def get_chat_messages(conv_id: int):
        try:
            conversation = Conversation.query.get(conv_id)
            chat_messages = conversation.chat_messages
            return json.loads(chat_messages)
        except Exception as e:
            print(e)
            return []

    def get_latest_conversation(user_id: int):
        try:
            user = User.query.get(user_id)
            conversations = user.conversations
            return conversations[-1].id if conversations != None and len(conversations) > 0 else None
        except Exception as e:
            print(e)


# class Sources(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     message_idx = db.Column(db.Integer, nullable=False)
#     sources = db.Column(db.Text, nullable=False)
#     conv_id = db.Column(db.Integer, db.ForeignKey("conversation.id"), nullable=False)

#     def save_observation(conv_id: int, message_idx: int, new_sources: List):
#         observation = Sources.query.filter(Sources.conv_id == conv_id, Sources.message_idx == message_idx).first()
#         if observation != None:
#             observation.observation = json.dumps(new_sources)
#         else:
#             observation = Sources(
#                 message_idx=message_idx,
#                 sources=json.dumps(new_sources),
#                 conv_id=conv_id,
#             )
#         db.session.add(observation)
#         db.session.commit()

#     def get_observation(conv_id: int, message_idx: int):
#         source = Sources.query.filter(Sources.conv_id == conv_id, Sources.message_idx == message_idx).first()
#         if source != None:
#             return json.loads(source.sources)
#         else:
#             return []

#     def delete_observation(conv_id: int, message_idx: int):
#         source = Sources.query.filter(Sources.conv_id == conv_id, Sources.message_idx == message_idx).first()
#         if source != None:
#             db.session.delete(source)
#             db.session.commit()

#     def delete_all_observations(conv_id: int):
#         sources = Sources.query.filter(Sources.conv_id == conv_id)
#         for source in sources:
#             db.session.delete(source)
#         db.session.commit()
