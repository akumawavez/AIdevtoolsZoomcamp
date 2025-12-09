import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { io } from 'socket.io-client';
import CodeEditor from '../components/CodeEditor';

const InterviewRoom = () => {
    const { sessionId } = useParams();
    const [code, setCode] = useState("// Start coding here");
    const [output, setOutput] = useState([]);
    const socketRef = useRef(null);

    useEffect(() => {
        // Connect to backend
        socketRef.current = io('http://localhost:3000');

        // Join room
        socketRef.current.emit('join-room', sessionId);

        // Listen for code updates
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

    const runCode = () => {
        setOutput([]); // Clear previous output
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
                setOutput(prev => [...prev, ...e.data.data]);
            } else if (e.data.type === 'error') {
                setOutput(prev => [...prev, `Error: ${e.data.data}`]);
            }
        };

        worker.onerror = (e) => {
            setOutput(prev => [...prev, `Worker Error: ${e.message}`]);
        }

        // Terminate worker after 5 seconds to prevent infinite loops
        setTimeout(() => {
            worker.terminate();
        }, 5000);
    };

    const [language, setLanguage] = useState("javascript");

    const handleLanguageChange = (e) => {
        setLanguage(e.target.value);
    };

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            <div style={{ flex: 1, borderRight: '1px solid #ccc' }}>
                <div style={{ padding: '10px', borderBottom: '1px solid #ccc', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>Room: {sessionId}</span>
                    <div>
                        <select value={language} onChange={handleLanguageChange} style={{ marginRight: '10px' }}>
                            <option value="javascript">JavaScript</option>
                            <option value="python">Python</option>
                        </select>
                        <button onClick={runCode} disabled={language !== 'javascript'}>Run Code</button>
                    </div>
                </div>
                <div style={{ height: 'calc(100% - 50px)' }}>
                    <CodeEditor code={code} setCode={handleCodeChange} language={language} />
                </div>
            </div>
            <div style={{ width: '300px', padding: '10px', backgroundColor: '#f0f0f0', overflowY: 'auto' }}>
                <h3>Output</h3>
                {output.map((line, index) => (
                    <div key={index} style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                        {typeof line === 'object' ? JSON.stringify(line) : line}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default InterviewRoom;
