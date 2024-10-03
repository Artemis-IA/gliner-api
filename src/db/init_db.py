# src/db/init_db.py
from db.base import Base
from db.session import engine
from db.models import Dataset, Inference, TrainingRun

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Base de données initialisée.")
