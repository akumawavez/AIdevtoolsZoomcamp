# Online Coding Interview Platform

A real-time collaborative coding platform with safe browser-based code execution.

## Prerequisites

- Node.js (v14 or higher)
- npm

## Project Structure

- `client`: React + Vite frontend application.
- `server`: Express.js + Socket.io backend server.

## Getting Started

### Quick Start

1.  **Install Dependencies**:
    Run the following command from the root directory to install dependencies for the root, client, and server:
    ```bash
    npm run install:all
    ```

2.  **Run the Application**:
    Start both the client and server concurrently:
    ```bash
    npm run dev
    ```

    - Frontend: `http://localhost:5173`
    - Backend: `http://localhost:3000`

## Usage

1. Open `http://localhost:5173` in your browser.
2. Click "Create New Interview" to generate a unique session link.
3. Share the URL with another user (or open in a new tab/window).
4. Collaborate on code in real-time.
5. Click "Run Code" to execute the JavaScript code safely in your browser.

## Testing

Integration tests are located in the `server` directory. They verify the real-time collaboration features.

To run the tests:

```bash
cd server
npm install # if not already installed
npx jest
```

## API Documentation

The session management API is defined in `openapi.yaml` in the root directory.

- `POST /api/sessions`: Create a new session.
- `GET /api/sessions/:sessionId`: Get session details.
