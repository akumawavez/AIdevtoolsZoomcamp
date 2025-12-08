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
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
            <h1>Online Coding Interview Platform</h1>
            <button onClick={createSession} style={{ padding: '10px 20px', fontSize: '1.2rem', cursor: 'pointer' }}>
                Create New Interview
            </button>
        </div>
    );
};

export default LandingPage;
