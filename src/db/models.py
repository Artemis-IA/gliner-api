# src/db/models
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime
from sqlalchemy.sql import func
from .base import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    data = Column(JSON, nullable=False)  # Stockage des données NER en JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Inference(Base):
    __tablename__ = "inferences"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    entities = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, nullable=False)
    dataset_id = Column(Integer, nullable=False)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=32)
    status = Column(String, default="Started")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
