import json
import os
from datetime import datetime
from typing import List

from flask_sqlalchemy import SQLAlchemy
from pydantic.errors import NoneIsNotAllowedError
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
    chat_messages_meta_data = db.Column(db.Text)
    thread_id = db.Column(db.Text)
    timestamp_last_message = db.Column(db.DateTime, nullable=False)

    def save_meta_data(conv_id: int, msg_idx: int, meta_data: dict):
        try:
            conversation = Conversation.query.get(conv_id)
            curr_meta_data = conversation.chat_messages_meta_data
            curr_meta_data = json.loads(curr_meta_data) if curr_meta_data else {}
            curr_meta_data.update({msg_idx: meta_data})
            conversation.chat_messages_meta_data = json.dumps(curr_meta_data)
            db.session.commit()
        except Exception as e:
            print(e)

    def generate_new_conversation(user_id: int, thread_id: str = None):
        conversation = Conversation(
            user_id=user_id, timestamp_last_message=datetime.now()
        )
        if not conversation:
            raise ValueError(f"No conversation found for user id {user_id}")
        if thread_id:
            conversation.thread_id = thread_id
        db.session.add(conversation)
        db.session.commit()
        return conversation.id

    def update_chat_messages(conv_id: int, chat_messages: List):
        conversation = Conversation.query.get(conv_id)
        if not conversation:
            raise ValueError(f"No conversation found for conv id {conv_id}")
        conversation.chat_messages = json.dumps(chat_messages)
        conversation.timestamp_last_message = datetime.now()
        db.session.commit()

    def get_chat_messages(conv_id: int) -> list[dict[str, str]]:
        conversation = Conversation.query.get(conv_id)
        if not conversation:
            raise ValueError(f"No conversation found for conv id {conv_id}")
        chat_messages = conversation.chat_messages
        return json.loads(chat_messages) if chat_messages else []

    def get_latest_conversation_id(user_id: int):
        user = User.query.get(user_id)
        conversations = user.conversations
        return (
            conversations[-1].id
            if conversations != None and len(conversations) > 0
            else None
        )

    def get_thread_id_from_conv_id(conv_id: int):
        conversation = Conversation.query.get(conv_id)
        if not conversation:
            raise ValueError(f"No conversation found for conv id {conv_id}")
        return conversation.thread_id

    def update_thread_id_from_conv_id(conv_id: int, thread_id: str):
        conversation = Conversation.query.get(conv_id)
        if not conversation:
            raise ValueError(f"No conversation found for conv id {conv_id}")
        conversation.thread_id = thread_id
        db.session.commit()


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_type = db.Column(db.Text, nullable=False)
    document_text = db.Column(db.Text, nullable=False)
    chapter_with_summaries = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def save_document(user_id: int, document_type: str, document_text: str):
        document = Document(
            user_id=user_id, document_type=document_type, document_text=document_text
        )
        db.session.add(document)
        db.session.commit()
        return document.id


class UserInformation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    information_type = db.Column(db.Text, nullable=False)
    information_text = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def save_user_information(user_id: int, information_text: str):
        pass
