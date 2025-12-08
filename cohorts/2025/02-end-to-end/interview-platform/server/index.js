const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*", // Allow all origins for simplicity
        methods: ["GET", "POST"]
    }
});

app.use(cors());
app.use(express.json());

// In-memory storage for sessions (replace with DB in production)
const sessions = {};

// API Routes
app.post('/api/sessions', (req, res) => {
    const sessionId = uuidv4();
    sessions[sessionId] = {
        createdAt: new Date(),
        code: "// Start coding here"
    };
    res.status(201).json({ sessionId });
});

app.get('/api/sessions/:sessionId', (req, res) => {
    const { sessionId } = req.params;
    if (sessions[sessionId]) {
        res.json({ sessionId, ...sessions[sessionId] });
    } else {
        res.status(404).json({ error: "Session not found" });
    }
});

// Socket.io Logic
io.on('connection', (socket) => {
    console.log('A user connected:', socket.id);

    socket.on('join-room', (roomId) => {
        socket.join(roomId);
        console.log(`User ${socket.id} joined room ${roomId}`);

        // Send current code if session exists
        if (sessions[roomId]) {
            socket.emit('code-update', sessions[roomId].code);
        }
    });

    socket.on('code-change', ({ roomId, code }) => {
        // Update session state
        if (sessions[roomId]) {
            sessions[roomId].code = code;
        } else {
            // Create session if it doesn't exist (e.g. direct link access)
            sessions[roomId] = { createdAt: new Date(), code };
        }

        // Broadcast to others in the room
        socket.to(roomId).emit('code-update', code);
    });

    socket.on('disconnect', () => {
        console.log('User disconnected:', socket.id);
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
