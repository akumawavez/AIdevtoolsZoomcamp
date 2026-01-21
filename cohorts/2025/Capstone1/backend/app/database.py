from sqlalchemy import create_engine, Column, Integer, String, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/aptitude_test")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Difficulty(str, enum.Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, enum.Enum):
    PYTHON = "python"
    ML_FUNDAMENTALS = "ml_fundamentals"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    MLOPS = "mlops"
    DATA_ENGINEERING = "data_engineering"
    MATH_STATS = "math_stats"
    LLM = "llm"


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    option_a = Column(String(500), nullable=False)
    option_b = Column(String(500), nullable=False)
    option_c = Column(String(500), nullable=False)
    option_d = Column(String(500), nullable=False)
    correct_option = Column(String(1), nullable=False)  # 'a', 'b', 'c', or 'd'
    difficulty = Column(SQLEnum(Difficulty), nullable=False)
    category = Column(SQLEnum(Category), nullable=False)
    explanation = Column(Text, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
