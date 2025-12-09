import React from 'react';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';

const LandingPage = () => {
    const navigate = useNavigate();

    const createSession = () => {
        const sessionId = uuidv4();
        navigate(`/room/${sessionId}`);
    };

    return (
        <div className="landing-container">
            <div className="hero-section">
                <div className="logo-badge">🚀 InterviewPlatform</div>
                <h1>Real-time Coding Interviews</h1>
                <p className="subtitle">
                    Collaborate on code in real-time with candidates.
                    Run JavaScript and Python directly in the browser.
                    No setup required.
                </p>

                <div className="cta-container">
                    <button onClick={createSession} className="btn btn-primary btn-lg">
                        Create New Interview
                    </button>
                </div>

                <div className="features-grid">
                    <div className="feature-card">
                        <span className="icon">⚡</span>
                        <h3>Real-time Sync</h3>
                        <p>See code changes instantly as you type.</p>
                    </div>
                    <div className="feature-card">
                        <span className="icon">🏃</span>
                        <h3>Browser Execution</h3>
                        <p>Run JS and Python code safely in the browser.</p>
                    </div>
                    <div className="feature-card">
                        <span className="icon">🎨</span>
                        <h3>Syntax Highlighting</h3>
                        <p>Beautiful editor with multi-language support.</p>
                    </div>
                </div>
            </div>

            <style>{`
        .landing-container {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
          padding: 2rem;
        }

        .hero-section {
          max-width: 800px;
          text-align: center;
        }

        .logo-badge {
          display: inline-block;
          padding: 0.5rem 1rem;
          background-color: rgba(0, 122, 204, 0.1);
          color: var(--accent-primary);
          border-radius: 50px;
          font-weight: 600;
          font-size: 0.9rem;
          margin-bottom: 1.5rem;
          border: 1px solid rgba(0, 122, 204, 0.2);
        }

        .subtitle {
          font-size: 1.2rem;
          margin: 1.5rem 0 2.5rem;
          max-width: 600px;
          margin-left: auto;
          margin-right: auto;
        }

        .btn-lg {
          padding: 1rem 2.5rem;
          font-size: 1.1rem;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 2rem;
          margin-top: 4rem;
        }

        .feature-card {
          background-color: var(--bg-secondary);
          padding: 1.5rem;
          border-radius: 12px;
          border: 1px solid var(--border-color);
          text-align: left;
          transition: transform 0.2s;
        }

        .feature-card:hover {
          transform: translateY(-5px);
          border-color: var(--accent-primary);
        }

        .feature-card .icon {
          font-size: 2rem;
          margin-bottom: 1rem;
          display: block;
        }
        
        .feature-card h3 {
           font-size: 1.1rem;
           margin-bottom: 0.5rem;
           color: var(--text-primary);
        }
        
        .feature-card p {
           font-size: 0.9rem;
           margin: 0;
        }
      `}</style>
        </div>
    );
};

export default LandingPage;
