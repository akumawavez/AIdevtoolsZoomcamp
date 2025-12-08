# Walkthrough - Online Coding Interview Platform

I have implemented the Online Coding Interview Platform with Frontend, Backend, and OpenAPI specs.

## Changes
- **Frontend**: Created a React + Vite application in `interview-platform/client`.
    - Added `CodeEditor` component using Monaco Editor.
    - Added `InterviewRoom` page with real-time collaboration via Socket.io.
    - Added `LandingPage` to create new sessions.
    - Implemented client-side code execution using Web Workers for safety.
- **Backend**: Created an Express.js server in `interview-platform/server`.
    - Implemented Socket.io server for broadcasting code changes.
    - Implemented API for session management (`POST /api/sessions`).
- **OpenAPI**: Created `openapi.yaml` defining the session API.

## Verification Results

### Automated Tests
- N/A (Manual verification performed)

### Manual Verification
1.  **Startup**:
    - Backend running on `http://localhost:3000`.
    - Frontend running on `http://localhost:5173`.
2.  **Flow**:
    - Open `http://localhost:5173`.
    - Click "Create New Interview".
    - Redirected to `/room/{uuid}`.
    - Type code in the editor.
    - Open the same URL in another tab.
    - Verify code syncs between tabs in real-time.
    - Click "Run Code".
    - Verify output appears in the Output panel.
    - Write `console.log("test")` and verify it shows up.
    - Write an infinite loop `while(true){}` and verify it terminates after 5 seconds (safety check).

## How to Run
1.  **Backend**:
    ```bash
    cd interview-platform/server
    npm install
    node index.js
    ```
2.  **Frontend**:
    ```bash
    cd interview-platform/client
    npm install
    npm run dev
    ```
