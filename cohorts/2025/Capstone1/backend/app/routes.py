from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from collections import defaultdict

from .database import get_db, Question, Difficulty, Category
from .schemas import QuestionOut, QuizSubmission, QuizResult, CategoryScore
from .mcp_client import analyze_quiz_performance, generate_study_plan

router = APIRouter(prefix="/api/questions", tags=["questions"])


@router.get("/", response_model=List[QuestionOut])
def get_questions(
    difficulty: Difficulty = None,
    category: Category = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    query = db.query(Question)
    if difficulty:
        query = query.filter(Question.difficulty == difficulty)
    if category:
        query = query.filter(Question.category == category)
    return query.limit(limit).all()


@router.get("/quiz", response_model=List[QuestionOut])
def get_adaptive_quiz(num_questions: int = 15, db: Session = Depends(get_db)):
    """Get questions with progressive difficulty (easy -> medium -> hard)"""
    easy_count = num_questions // 3
    medium_count = num_questions // 3
    hard_count = num_questions - easy_count - medium_count

    easy = db.query(Question).filter(Question.difficulty == Difficulty.EASY).limit(easy_count).all()
    medium = db.query(Question).filter(Question.difficulty == Difficulty.MEDIUM).limit(medium_count).all()
    hard = db.query(Question).filter(Question.difficulty == Difficulty.HARD).limit(hard_count).all()

    return easy + medium + hard


@router.post("/submit")
def submit_quiz(submission: QuizSubmission, db: Session = Depends(get_db)):
    """Evaluate quiz submission with MCP-powered analysis and provide detailed feedback"""
    question_ids = [a.question_id for a in submission.answers]
    questions = db.query(Question).filter(Question.id.in_(question_ids)).all()
    question_map = {q.id: q for q in questions}

    correct = 0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    difficulty_stats = {"easy": 0, "medium": 0, "hard": 0}

    for answer in submission.answers:
        q = question_map.get(answer.question_id)
        if not q:
            continue
        category_stats[q.category.value]["total"] += 1
        if answer.selected_option.lower() == q.correct_option.lower():
            correct += 1
            category_stats[q.category.value]["correct"] += 1
            difficulty_stats[q.difficulty.value] += 1

    total = len(submission.answers)
    score_pct = (correct / total * 100) if total > 0 else 0

    category_scores = []
    strengths = []
    areas_to_improve = []

    category_names = {
        "python": "Python Programming",
        "ml_fundamentals": "ML Fundamentals",
        "deep_learning": "Deep Learning",
        "nlp": "Natural Language Processing",
        "mlops": "MLOps & Deployment",
        "data_engineering": "Data Engineering",
        "math_stats": "Math & Statistics",
        "llm": "Large Language Models"
    }

    category_scores_for_mcp = []
    
    for cat, stats in category_stats.items():
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        cat_name = category_names.get(cat, cat)
        category_scores.append(CategoryScore(
            category=cat_name,
            correct=stats["correct"],
            total=stats["total"],
            percentage=round(pct, 1)
        ))
        category_scores_for_mcp.append({
            "category": cat_name,
            "correct": stats["correct"],
            "total": stats["total"],
            "percentage": round(pct, 1)
        })
        if pct >= 80:
            strengths.append(cat_name)
        elif pct < 50:
            areas_to_improve.append(cat_name)

    # Call MCP evaluation tool for enhanced analysis
    mcp_analysis = analyze_quiz_performance(
        category_scores=category_scores_for_mcp,
        total_score=score_pct,
        difficulty_breakdown=difficulty_stats
    )

    # Generate study plan if there are weak areas
    study_plan = None
    if areas_to_improve:
        study_plan = generate_study_plan(
            weak_categories=areas_to_improve,
            skill_level=mcp_analysis.get("estimated_skill_level", "Intermediate")
        )

    # Build enhanced feedback from MCP analysis
    feedback = mcp_analysis.get("personalized_feedback", [])
    if not feedback:
        if score_pct >= 80:
            feedback.append("Excellent performance! You have a strong grasp of AI Engineering concepts.")
        elif score_pct >= 60:
            feedback.append("Good job! You have solid foundational knowledge.")
        else:
            feedback.append("Keep learning! Focus on the areas highlighted below.")

    return {
        "total_questions": total,
        "correct_answers": correct,
        "score_percentage": round(score_pct, 1),
        "category_scores": category_scores,
        "feedback": feedback,
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "mcp_analysis": {
            "performance_level": mcp_analysis.get("performance_level", ""),
            "estimated_skill_level": mcp_analysis.get("estimated_skill_level", ""),
            "study_recommendations": mcp_analysis.get("study_recommendations", []),
            "next_steps": mcp_analysis.get("next_steps", [])
        },
        "study_plan": study_plan
    }


@router.get("/mcp-status")
def get_mcp_status():
    """Check if MCP tools are available"""
    return {
        "mcp_enabled": True,
        "available_tools": [
            "analyze_quiz_performance",
            "generate_study_plan",
            "get_question_explanation"
        ],
        "description": "MCP server provides intelligent analysis and personalized feedback"
    }
