# src/db/init_db.py
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import sessionmaker
from db.session import engine
from db.models import Base

def init_db():
    # Create tables if they don't exist (for legacy usage)
    Base.metadata.create_all(bind=engine)

    # Run migrations using Alembic
    alembic_cfg = Config("alembic.ini") 
    command.upgrade(alembic_cfg, "head")  # Run all migrations
