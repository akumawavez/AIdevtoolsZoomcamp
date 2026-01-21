import { useState, useEffect } from 'react'
import './App.css'

interface Feature {
  icon: string
  title: string
  description: string
}

interface Question {
  id: number
  text: string
  option_a: string
  option_b: string
  option_c: string
  option_d: string
  difficulty: string
  category: string
}

interface CategoryScore {
  category: string
  correct: number
  total: number
  percentage: number
}

interface MCPAnalysis {
  performance_level: string
  estimated_skill_level: string
  study_recommendations: string[]
  next_steps: string[]
}

interface StudyPlan {
  weekly_goals: string[]
  recommended_resources: Record<string, string[]>
  practice_projects: string[]
  timeline: string
}

interface QuizResult {
  total_questions: number
  correct_answers: number
  score_percentage: number
  category_scores: CategoryScore[]
  feedback: string[]
  strengths: string[]
  areas_to_improve: string[]
  mcp_analysis?: MCPAnalysis
  study_plan?: StudyPlan
}

const features: Feature[] = [
  {
    icon: '🤖',
    title: 'AI-Generated Tests',
    description: 'Generate tailored MCQs, coding challenges, and logic puzzles based on role and skills.'
  },
  {
    icon: '⚡',
    title: 'Instant Evaluation',
    description: 'AI-powered grading with detailed feedback and skill breakdown in seconds.'
  },
  {
    icon: '📊',
    title: 'Smart Analytics',
    description: 'Visualize candidate performance with AI-driven insights and rankings.'
  }
]

const API_URL = 'http://localhost:8000'

function App() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const [showDemo, setShowDemo] = useState(false)
  const [questions, setQuestions] = useState<Question[]>([])
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null)
  const [answers, setAnswers] = useState<{ question_id: number, selected_option: string }[]>([])
  const [result, setResult] = useState<QuizResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [showStudyPlan, setShowStudyPlan] = useState(false)

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`)
      const data = await response.json()
      setIsHealthy(data.status === 'healthy')
    } catch {
      setIsHealthy(false)
    }
  }

  useEffect(() => {
    checkHealth()
  }, [])

  const startDemo = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/questions/quiz?num_questions=15`)
      const data = await response.json()
      setQuestions(data)
      setShowDemo(true)
      setCurrentQuestion(0)
      setSelectedAnswer(null)
      setAnswers([])
      setResult(null)
      setShowStudyPlan(false)
    } catch (error) {
      console.error('Failed to fetch questions:', error)
    }
    setLoading(false)
  }

  const handleAnswer = (option: string) => {
    setSelectedAnswer(option)
  }

  const nextQuestion = () => {
    if (selectedAnswer && questions[currentQuestion]) {
      const newAnswers = [...answers, {
        question_id: questions[currentQuestion].id,
        selected_option: selectedAnswer
      }]
      setAnswers(newAnswers)

      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1)
        setSelectedAnswer(null)
      } else {
        submitQuiz(newAnswers)
      }
    }
  }

  const submitQuiz = async (finalAnswers: { question_id: number, selected_option: string }[]) => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/api/questions/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: finalAnswers })
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Failed to submit quiz:', error)
    }
    setLoading(false)
  }

  const closeDemo = () => {
    setShowDemo(false)
    setResult(null)
    setShowStudyPlan(false)
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return '#10b981'
      case 'medium': return '#f59e0b'
      case 'hard': return '#ef4444'
      default: return '#6366f1'
    }
  }

  return (
    <div className="app">
      {/* Demo Modal */}
      {showDemo && (
        <div className="modal-overlay" onClick={closeDemo}>
          <div className="modal modal-large" onClick={(e: React.MouseEvent) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeDemo}>×</button>

            {loading ? (
              <div className="loading">Loading...</div>
            ) : result ? (
              <div className="results">
                {!showStudyPlan ? (
                  <>
                    <div className="results-header">
                      <div className="results-icon">
                        {result.score_percentage >= 80 ? '🏆' : result.score_percentage >= 50 ? '👍' : '📚'}
                      </div>
                      <h3>Quiz Complete!</h3>
                      <p className="score">{result.correct_answers}/{result.total_questions} ({result.score_percentage}%)</p>

                      {result.mcp_analysis && (
                        <div className="skill-badge">
                          <span className="level">{result.mcp_analysis.performance_level}</span>
                          <span className="skill">{result.mcp_analysis.estimated_skill_level}</span>
                        </div>
                      )}
                    </div>

                    {result.feedback.map((f: string, i: number) => (
                      <p key={i} className="feedback">{f}</p>
                    ))}

                    <div className="category-breakdown">
                      <h4>Performance by Category</h4>
                      {result.category_scores.map((cs, i) => (
                        <div key={i} className="category-row">
                          <span className="category-name">{cs.category}</span>
                          <div className="category-bar">
                            <div
                              className="category-fill"
                              style={{
                                width: `${cs.percentage}%`,
                                background: cs.percentage >= 80 ? '#10b981' : cs.percentage >= 50 ? '#f59e0b' : '#ef4444'
                              }}
                            />
                          </div>
                          <span className="category-percent">{cs.percentage}%</span>
                        </div>
                      ))}
                    </div>

                    {result.strengths.length > 0 && (
                      <div className="strengths">
                        <h4>💪 Strengths</h4>
                        <p>{result.strengths.join(', ')}</p>
                      </div>
                    )}

                    {result.areas_to_improve.length > 0 && (
                      <div className="improve">
                        <h4>📖 Areas to Improve</h4>
                        <p>{result.areas_to_improve.join(', ')}</p>
                      </div>
                    )}

                    {result.mcp_analysis && result.mcp_analysis.next_steps.length > 0 && (
                      <div className="next-steps">
                        <h4>🎯 Recommended Next Steps</h4>
                        <ul>
                          {result.mcp_analysis.next_steps.map((step: string, i: number) => (
                            <li key={i}>{step}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="action-buttons">
                      {result.study_plan && (
                        <button className="btn-secondary btn-large" onClick={() => setShowStudyPlan(true)}>
                          📚 View Study Plan
                        </button>
                      )}
                      <button className="btn-primary btn-large" onClick={closeDemo}>
                        Back to Home
                      </button>
                    </div>
                  </>
                ) : result.study_plan && (
                  <>
                    <div className="study-plan">
                      <h3>📚 Your Personalized Study Plan</h3>
                      <p className="timeline">{result.study_plan.timeline}</p>

                      <div className="weekly-goals">
                        <h4>Weekly Goals</h4>
                        <ul>
                          {result.study_plan.weekly_goals.map((goal: string, i: number) => (
                            <li key={i}>{goal}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="resources">
                        <h4>Recommended Resources</h4>
                        {Object.entries(result.study_plan.recommended_resources).map(([cat, resources]) => (
                          <div key={cat} className="resource-category">
                            <h5>{cat}</h5>
                            <ul>
                              {resources.map((r: string, i: number) => (
                                <li key={i}>{r}</li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>

                      <div className="projects">
                        <h4>Practice Projects</h4>
                        <ul>
                          {result.study_plan.practice_projects.map((p: string, i: number) => (
                            <li key={i}>{p}</li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="action-buttons">
                      <button className="btn-secondary btn-large" onClick={() => setShowStudyPlan(false)}>
                        ← Back to Results
                      </button>
                      <button className="btn-primary btn-large" onClick={closeDemo}>
                        Done
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : questions.length > 0 ? (
              <>
                <div className="question-header">
                  <div className="question-meta">
                    <span className="question-number">Question {currentQuestion + 1}/{questions.length}</span>
                    <span
                      className="difficulty-badge"
                      style={{ background: getDifficultyColor(questions[currentQuestion].difficulty) }}
                    >
                      {questions[currentQuestion].difficulty}
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                    />
                  </div>
                </div>

                <h3 className="question-text">{questions[currentQuestion].text}</h3>

                <div className="options">
                  {['a', 'b', 'c', 'd'].map((opt) => (
                    <button
                      key={opt}
                      className={`option ${selectedAnswer === opt ? 'selected' : ''}`}
                      onClick={() => handleAnswer(opt)}
                    >
                      <span className="option-letter">{opt.toUpperCase()}</span>
                      {questions[currentQuestion][`option_${opt}` as keyof Question]}
                    </button>
                  ))}
                </div>

                <button
                  className="btn-primary btn-large"
                  onClick={nextQuestion}
                  disabled={selectedAnswer === null}
                  style={{ marginTop: '1.5rem', width: '100%', opacity: selectedAnswer === null ? 0.5 : 1 }}
                >
                  {currentQuestion < questions.length - 1 ? 'Next Question' : 'Submit Quiz'}
                </button>
              </>
            ) : (
              <div className="loading">No questions available. Make sure the database is seeded.</div>
            )}
          </div>
        </div>
      )}

      {/* Hero Section */}
      <header className="hero">
        <nav className="nav">
          <div className="logo">🧠 AI Aptitude</div>
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#demo">Demo</a>
            <button className="btn-primary" onClick={checkHealth}>
              Check Backend
            </button>
          </div>
        </nav>

        <div className="hero-content">
          <h1>AI-Powered Aptitude Testing Platform</h1>
          <p className="hero-subtitle">
            Automate technical hiring with AI-generated tests and intelligent evaluation.
            Save 80% of your screening time.
          </p>
          <div className="hero-actions">
            <button className="btn-primary btn-large" onClick={startDemo} disabled={loading}>
              {loading ? 'Loading...' : 'Get Started'}
            </button>
            <button className="btn-secondary btn-large" onClick={startDemo} disabled={loading}>
              View Demo
            </button>
          </div>

          {isHealthy !== null && (
            <div className={`health-status ${isHealthy ? 'healthy' : 'unhealthy'}`}>
              Backend: {isHealthy ? '✅ Connected' : '❌ Disconnected'}
            </div>
          )}
        </div>
      </header>

      {/* Features Section */}
      <section id="features" className="features-section">
        <h2>Why Choose AI Aptitude?</h2>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="demo-section">
        <h2>Try a Sample Test</h2>
        <div className="demo-card">
          <p>Experience our AI-powered testing platform with a 15-question adaptive quiz covering AI Engineering topics.</p>
          <button className="btn-primary" onClick={startDemo} disabled={loading}>
            {loading ? 'Loading...' : 'Start Demo Test'}
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>🚀 Built for AI Engineering Zoomcamp 2025</p>
      </footer>
    </div>
  )
}

export default App
