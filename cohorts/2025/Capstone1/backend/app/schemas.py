from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    PYTHON = "python"
    ML_FUNDAMENTALS = "ml_fundamentals"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    MLOPS = "mlops"
    DATA_ENGINEERING = "data_engineering"
    MATH_STATS = "math_stats"
    LLM = "llm"


class QuestionOut(BaseModel):
    id: int
    text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    difficulty: Difficulty
    category: Category

    class Config:
        from_attributes = True


class QuestionWithAnswer(QuestionOut):
    correct_option: str
    explanation: Optional[str] = None


class AnswerSubmission(BaseModel):
    question_id: int
    selected_option: str


class QuizSubmission(BaseModel):
    answers: List[AnswerSubmission]


class CategoryScore(BaseModel):
    category: str
    correct: int
    total: int
    percentage: float


class QuizResult(BaseModel):
    total_questions: int
    correct_answers: int
    score_percentage: float
    category_scores: List[CategoryScore]
    feedback: List[str]
    strengths: List[str]
    areas_to_improve: List[str]
