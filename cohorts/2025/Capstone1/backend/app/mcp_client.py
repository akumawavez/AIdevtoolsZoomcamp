"""
MCP Client service for calling the evaluation tools.

This module provides functions to call MCP tools directly
without running a separate server process (in-process usage).
"""
from typing import List, Dict, Any


def analyze_quiz_performance(
    category_scores: List[Dict[str, Any]],
    total_score: float,
    difficulty_breakdown: Dict[str, int]
) -> Dict[str, Any]:
    """
    Analyzes quiz performance and generates personalized feedback.
    
    This is a direct implementation matching the MCP tool interface.
    """
    analysis = {
        "performance_level": "",
        "personalized_feedback": [],
        "study_recommendations": [],
        "next_steps": [],
        "estimated_skill_level": ""
    }
    
    # Determine overall performance level
    if total_score >= 85:
        analysis["performance_level"] = "Expert"
        analysis["estimated_skill_level"] = "Senior/Lead Level"
    elif total_score >= 70:
        analysis["performance_level"] = "Proficient"
        analysis["estimated_skill_level"] = "Mid-Level"
    elif total_score >= 50:
        analysis["performance_level"] = "Intermediate"
        analysis["estimated_skill_level"] = "Junior Level"
    else:
        analysis["performance_level"] = "Beginner"
        analysis["estimated_skill_level"] = "Entry Level"
    
    # Analyze category performance
    weak_categories = []
    strong_categories = []
    
    for cat in category_scores:
        if cat["percentage"] < 50:
            weak_categories.append(cat["category"])
            analysis["study_recommendations"].append(
                f"Focus on {cat['category']}: You scored {cat['percentage']}%. "
                f"Review fundamentals and practice more problems."
            )
        elif cat["percentage"] >= 80:
            strong_categories.append(cat["category"])
    
    # Generate personalized feedback
    if strong_categories:
        analysis["personalized_feedback"].append(
            f"Strong performance in: {', '.join(strong_categories)}. "
            "Consider mentoring others or exploring advanced topics."
        )
    
    if weak_categories:
        analysis["personalized_feedback"].append(
            f"Areas needing attention: {', '.join(weak_categories)}. "
            "Dedicate focused study time to these topics."
        )
    
    # Analyze difficulty progression
    easy = difficulty_breakdown.get("easy", 0)
    medium = difficulty_breakdown.get("medium", 0)
    hard = difficulty_breakdown.get("hard", 0)
    
    if hard > medium > easy:
        analysis["personalized_feedback"].append(
            "Interesting pattern: You performed better on harder questions. "
            "You might be overthinking simpler problems."
        )
    elif easy > medium > hard:
        analysis["personalized_feedback"].append(
            "Good foundation with easier concepts. "
            "Push yourself with more challenging problems to grow."
        )
    
    # Generate next steps
    analysis["next_steps"] = [
        "Review incorrect answers and understand why",
        "Practice 5-10 problems daily in weak areas",
        "Build mini-projects to apply concepts",
        "Join AI/ML communities for peer learning"
    ]
    
    return analysis


def generate_study_plan(
    weak_categories: List[str],
    skill_level: str
) -> Dict[str, Any]:
    """
    Generates a personalized study plan based on identified weaknesses.
    """
    resources = {
        "Python Programming": [
            "Python for Data Science (freeCodeCamp)",
            "Automate the Boring Stuff with Python",
            "LeetCode Python problems"
        ],
        "ML Fundamentals": [
            "Andrew Ng's Machine Learning Course",
            "Hands-On ML with Scikit-Learn (O'Reilly)",
            "Kaggle Learn ML courses"
        ],
        "Deep Learning": [
            "Deep Learning Specialization (Coursera)",
            "Fast.ai Practical Deep Learning",
            "PyTorch/TensorFlow tutorials"
        ],
        "Natural Language Processing": [
            "Hugging Face NLP Course",
            "Speech and Language Processing (Jurafsky)",
            "NLP with Transformers (O'Reilly)"
        ],
        "MLOps & Deployment": [
            "MLOps Zoomcamp (DataTalks)",
            "Made With ML MLOps course",
            "Kubernetes for ML deployment"
        ],
        "Data Engineering": [
            "Data Engineering Zoomcamp",
            "Designing Data-Intensive Applications",
            "Apache Spark documentation"
        ],
        "Math & Statistics": [
            "Khan Academy Statistics",
            "3Blue1Brown Linear Algebra",
            "StatQuest YouTube channel"
        ],
        "Large Language Models": [
            "LLM University by Cohere",
            "Prompt Engineering Guide",
            "Hugging Face LLM course"
        ]
    }
    
    plan = {
        "weekly_goals": [],
        "recommended_resources": {},
        "practice_projects": [],
        "timeline": ""
    }
    
    # Set timeline based on skill level
    if skill_level in ["Entry Level", "Beginner"]:
        plan["timeline"] = "3-6 months intensive study recommended"
    elif skill_level in ["Junior Level", "Intermediate"]:
        plan["timeline"] = "1-3 months focused practice recommended"
    else:
        plan["timeline"] = "2-4 weeks targeted review recommended"
    
    # Generate weekly goals
    for i, cat in enumerate(weak_categories[:4], 1):
        plan["weekly_goals"].append(f"Week {i}: Focus on {cat} fundamentals")
        plan["recommended_resources"][cat] = resources.get(cat, ["General AI/ML courses"])
    
    # Add practice projects
    plan["practice_projects"] = [
        "Build an end-to-end ML pipeline",
        "Create a simple chatbot with LLM API",
        "Deploy a model to cloud",
        "Contribute to an open-source AI project"
    ]
    
    return plan
