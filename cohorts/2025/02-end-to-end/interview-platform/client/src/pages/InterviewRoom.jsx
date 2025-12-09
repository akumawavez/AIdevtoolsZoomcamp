import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { io } from 'socket.io-client';
import CodeEditor from '../components/CodeEditor';

const InterviewRoom = () => {
    const { sessionId } = useParams();
    const [code, setCode] = useState("// Start coding here");
    const [output, setOutput] = useState([]);
    const socketRef = useRef(null);

    // Socket Setup
    useEffect(() => {
        socketRef.current = io('http://localhost:3000');
        socketRef.current.emit('join-room', sessionId);
        socketRef.current.on('code-update', (newCode) => {
            setCode(newCode);
        });
        return () => {
            socketRef.current.disconnect();
        };
    }, [sessionId]);

    const handleCodeChange = (newCode) => {
        setCode(newCode);
        socketRef.current.emit('code-change', { roomId: sessionId, code: newCode });
    };

    // Code Execution Setup
    const [language, setLanguage] = useState("javascript");
    const pyodideRef = useRef(null);
    const [isPyodideLoading, setIsPyodideLoading] = useState(true);

    useEffect(() => {
        const loadPyodideInstance = async () => {
            try {
                if (window.loadPyodide) {
                    pyodideRef.current = await window.loadPyodide();
                    setIsPyodideLoading(false);
                }
            } catch (error) {
                console.error("Failed to load Pyodide:", error);
            }
        };
        loadPyodideInstance();
    }, []);

    const handleLanguageChange = (e) => {
        const newLanguage = e.target.value;
        setLanguage(newLanguage);
        if (newLanguage === 'python') {
            setCode("# Start coding here\nprint('Hello World!')");
        } else {
            setCode("// Start coding here\nconsole.log('Hello World!');");
        }
    };

    const runCode = async () => {
        setOutput([{ type: 'system', content: 'Running...' }]);

        if (language === 'javascript') {
            const workerCode = `
              const originalConsoleLog = console.log;
              console.log = (...args) => {
                postMessage({ type: 'log', data: args });
                originalConsoleLog(...args);
              };
              
              try {
                ${code}
              } catch (error) {
                postMessage({ type: 'error', data: error.toString() });
              }
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const worker = new Worker(URL.createObjectURL(blob));

            worker.onmessage = (e) => {
                if (e.data.type === 'log') {
                    // Flatten args and add to output
                    const logs = e.data.data.map(arg =>
                        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                    );
                    setOutput(prev => [...prev.filter(p => p.type !== 'system'), ...logs.map(l => ({ type: 'log', content: l }))]);
                } else if (e.data.type === 'error') {
                    setOutput(prev => [...prev.filter(p => p.type !== 'system'), { type: 'error', content: e.data.data }]);
                }
            };

            worker.onerror = (e) => {
                setOutput(prev => [...prev.filter(p => p.type !== 'system'), { type: 'error', content: e.message }]);
            }

            setTimeout(() => {
                worker.terminate();
            }, 5000);

        } else if (language === 'python') {
            if (!pyodideRef.current) {
                setOutput([{ type: 'error', content: "Pyodide is still loading..." }]);
                return;
            }

            try {
                pyodideRef.current.setStdout({
                    batched: (msg) => setOutput(prev => [...prev.filter(p => p.type !== 'system'), { type: 'log', content: msg }])
                });
                await pyodideRef.current.runPythonAsync(code);
                // If we get here, execution finished without error
                if (output.length === 1 && output[0].type === 'system') {
                    // remove 'running' if there was no output
                    setOutput(prev => prev.filter(p => p.type !== 'system'));
                }
            } catch (error) {
                setOutput(prev => [...prev.filter(p => p.type !== 'system'), { type: 'error', content: error.message }]);
            }
        }
    };

    const clearOutput = () => {
        setOutput([]);
    }

    return (
        <div className="interview-container">
            {/* Header */}
            <header className="room-header">
                <div className="left-section">
                    <Link to="/" style={{ textDecoration: 'none' }}>
                        <div className="logo-small">🚀</div>
                    </Link>
                    <div className="room-info">
                        <span className="label">Session ID:</span>
                        <span className="value">{sessionId}</span>
                    </div>
                </div>

                <div className="right-section flex-row items-center gap-4">
                    <select
                        value={language}
                        onChange={handleLanguageChange}
                        className="select-input"
                    >
                        <option value="javascript">JavaScript</option>
                        <option value="python">Python</option>
                    </select>

                    <button
                        onClick={runCode}
                        className="btn btn-primary"
                        disabled={language === 'python' && isPyodideLoading}
                    >
                        {language === 'python' && isPyodideLoading ? "Loading..." : "Run Code ▶"}
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <div className="workspace">
                <div className="editor-pane">
                    <CodeEditor code={code} setCode={handleCodeChange} language={language} />
                </div>

                <div className="output-pane">
                    <div className="pane-header">
                        <span>Output</span>
                        <button className="clear-btn" onClick={clearOutput}>Clear</button>
                    </div>
                    <div className="terminal-window">
                        {output.length === 0 && <div className="placeholder">Run your code to see the output here...</div>}
                        {output.map((line, index) => (
                            <div key={index} className={`log-line ${line.type}`}>
                                {line.content}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <style>{`
                .interview-container {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                    background-color: var(--bg-primary);
                }

                .room-header {
                    height: 60px;
                    background-color: var(--bg-secondary);
                    border-bottom: 1px solid var(--border-color);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0 1.5rem;
                }

                .logo-small {
                    font-size: 1.5rem;
                    margin-right: 1rem;
                }

                .room-info {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                    background-color: var(--bg-tertiary);
                    padding: 0.25rem 0.75rem;
                    border-radius: 4px;
                }

                .room-info .value {
                    color: var(--text-primary);
                    font-family: monospace;
                }

                .workspace {
                    flex: 1;
                    display: flex;
                    overflow: hidden;
                }

                .editor-pane {
                    flex: 2; /* 66% width */
                    border-right: 1px solid var(--border-color);
                }

                .output-pane {
                    flex: 1; /* 33% width */
                    display: flex;
                    flex-direction: column;
                    background-color: #1e1e1e; /* Terminal background */
                }

                .pane-header {
                    height: 40px;
                    background-color: var(--bg-secondary);
                    border-bottom: 1px solid var(--border-color);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0 1rem;
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: var(--text-secondary);
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                
                .clear-btn {
                    background: none;
                    border: none;
                    color: var(--text-secondary);
                    cursor: pointer;
                    font-size: 0.8rem;
                }
                .clear-btn:hover { color: var(--text-primary); }

                .terminal-window {
                    flex: 1;
                    padding: 1rem;
                    overflow-y: auto;
                    font-family: 'Fira Code', 'Consolas', monospace;
                    font-size: 0.9rem;
                    line-height: 1.5;
                }

                .log-line {
                    white-space: pre-wrap;
                    word-break: break-all;
                    margin-bottom: 0.25rem;
                }

                .log-line.log { color: #cccccc; }
                .log-line.error { color: #f44336; }
                .log-line.system { color: #007acc; font-style: italic; }

                .placeholder {
                    color: #555;
                    font-style: italic;
                    text-align: center;
                    margin-top: 2rem;
                }
            `}</style>
        </div>
    );
};

export default InterviewRoom;
