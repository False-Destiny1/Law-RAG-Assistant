"""SQLAlchemy ORM models for the legal assistant application."""

from datetime import datetime, timezone

import bcrypt
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    phone = Column(String(20), unique=True, nullable=False)
    username = Column(String(50), nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    chats = relationship("Chat", backref="user", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", backref="user", cascade="all, delete-orphan")
    uploaded_documents = relationship("UploadedDocument", backref="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), self.password_hash.encode("utf-8"))


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )
    documents = relationship("UploadedDocument", backref="knowledge_base", cascade="all, delete-orphan")
    chats = relationship("Chat", backref="knowledge_base")


class Chat(Base):
    __tablename__ = "chat"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    title = Column(String(100), nullable=False, default="新对话")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=True)
    messages = relationship("Message", backref="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "message"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey("chat.id"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UploadedDocument(Base):
    __tablename__ = "uploaded_document"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    message_id = Column(Integer, nullable=False)
    chat_id = Column(Integer, nullable=False)
    rating = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class InterventionRequest(Base):
    __tablename__ = "intervention_request"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    chat_id = Column(Integer, ForeignKey("chat.id"), nullable=False)
    original_query = Column(Text, nullable=False)
    confidence_level = Column(String(20), nullable=False)
    confidence_score = Column(Float, nullable=False)
    confidence_reason = Column(Text)
    retrieved_doc_count = Column(Integer, default=0)
    status = Column(String(20), nullable=False, default="pending")
    assigned_to = Column(Integer, ForeignKey("user.id"), nullable=True)
    response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))


class KnowledgeGap(Base):
    __tablename__ = "knowledge_gap"
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    confidence_level = Column(String(20))
    confidence_reason = Column(Text)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    frequency = Column(Integer, default=1)
    status = Column(String(20), nullable=False, default="open")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))
