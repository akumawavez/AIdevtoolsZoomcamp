import { useState } from 'react'
import './App.css'

interface Feature {
  icon: string
  title: string
  description: string
}

interface Question {
  id: number
  text: string
  options: string[]
  correctIndex: number
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

const demoQuestions: Question[] = [
  {
    id: 1,
    text: 'What is the time complexity of binary search?',
    options: ['O(n)', 'O(log n)', 'O(n²)', 'O(1)'],
    correctIndex: 1
  },
  {
    id: 2,
    text: 'Which data structure uses LIFO (Last In, First Out)?',
    options: ['Queue', 'Array', 'Stack', 'Linked List'],
    correctIndex: 2
  },
  {
    id: 3,
    text: 'What does SQL stand for?',
    options: ['Structured Query Language', 'Simple Query Logic', 'System Query Language', 'Standard Query Library'],
    correctIndex: 0
  }
]

function App() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const [showDemo, setShowDemo] = useState(false)
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [score, setScore] = useState(0)
  const [showResults, setShowResults] = useState(false)

  const checkHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      const data = await response.json()
      setIsHealthy(data.status === 'healthy')
    } catch {
      setIsHealthy(false)
    }
  }

  const startDemo = () => {
    setShowDemo(true)
    setCurrentQuestion(0)
    setSelectedAnswer(null)
    setScore(0)
    setShowResults(false)
  }

  const handleAnswer = (index: number) => {
    setSelectedAnswer(index)
  }

  const nextQuestion = () => {
    if (selectedAnswer === demoQuestions[currentQuestion].correctIndex) {
      setScore(score + 1)
    }

    if (currentQuestion < demoQuestions.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
      setSelectedAnswer(null)
    } else {
      setShowResults(true)
    }
  }

  const closeDemo = () => {
    setShowDemo(false)
  }

  return (
    <div className="app">
      {/* Demo Modal */}
      {showDemo && (
        <div className="modal-overlay" onClick={closeDemo}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeDemo}>×</button>

            {!showResults ? (
              <>
                <div className="question-header">
                  <span className="question-number">Question {currentQuestion + 1}/{demoQuestions.length}</span>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${((currentQuestion + 1) / demoQuestions.length) * 100}%` }}
                    />
                  </div>
                </div>

                <h3 className="question-text">{demoQuestions[currentQuestion].text}</h3>

                <div className="options">
                  {demoQuestions[currentQuestion].options.map((option, index) => (
                    <button
                      key={index}
                      className={`option ${selectedAnswer === index ? 'selected' : ''}`}
                      onClick={() => handleAnswer(index)}
                    >
                      {option}
                    </button>
                  ))}
                </div>

                <button
                  className="btn-primary btn-large"
                  onClick={nextQuestion}
                  disabled={selectedAnswer === null}
                  style={{ marginTop: '1.5rem', width: '100%', opacity: selectedAnswer === null ? 0.5 : 1 }}
                >
                  {currentQuestion < demoQuestions.length - 1 ? 'Next Question' : 'Submit'}
                </button>
              </>
            ) : (
              <div className="results">
                <div className="results-icon">🎉</div>
                <h3>Demo Complete!</h3>
                <p className="score">You scored {score}/{demoQuestions.length}</p>
                <p className="feedback">
                  {score === demoQuestions.length
                    ? 'Perfect! You nailed it!'
                    : score >= demoQuestions.length / 2
                      ? 'Good job! Keep practicing.'
                      : 'Keep learning, you\'ll get better!'}
                </p>
                <button className="btn-primary btn-large" onClick={closeDemo} style={{ marginTop: '1.5rem' }}>
                  Back to Home
                </button>
              </div>
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
            <button className="btn-primary btn-large" onClick={startDemo}>Get Started</button>
            <button className="btn-secondary btn-large" onClick={startDemo}>View Demo</button>
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
          <p>Experience our AI-powered testing platform with a quick 3-question demo.</p>
          <button className="btn-primary" onClick={startDemo}>Start Demo Test</button>
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
