import { useState } from 'react'
import './App.css'

interface Feature {
  icon: string
  title: string
  description: string
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

function App() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)

  const checkHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      const data = await response.json()
      setIsHealthy(data.status === 'healthy')
    } catch {
      setIsHealthy(false)
    }
  }

  return (
    <div className="app">
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
            <button className="btn-primary btn-large">Get Started</button>
            <button className="btn-secondary btn-large">View Demo</button>
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
          <p>Experience our AI-powered testing platform with a quick demo.</p>
          <button className="btn-primary">Start Demo Test</button>
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
