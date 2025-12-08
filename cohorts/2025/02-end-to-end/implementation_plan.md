# Implementation Plan - Online Coding Interview Platform

This plan outlines the steps to build a real-time collaborative coding platform with safe browser-based code execution.

## User Review Required
> [!IMPORTANT]
> **Code Execution Strategy**: The requirement is to "Execute code safely in the browser". I plan to use a Web Worker based approach to run JavaScript code in a separate thread to prevent UI freezing and provide a basic level of isolation. For other languages, we might need a WASM-based solution or a backend service, but for this MVP, I will focus on JavaScript/TypeScript support using browser capabilities. If other languages are strictly required *in the browser*, we would need to integrate something like Pyodide (Python) etc., which adds significant weight. **Please confirm if JS/TS only is acceptable for the "browser execution" requirement, or if you need multi-language support via WASM/Backend.**
> *Assumption*: I will proceed with JavaScript/TypeScript support primarily for the browser execution demo, as it's native to the browser.

## Proposed Changes

### Structure
I will create a new directory `interview-platform` with two subdirectories:
- `client`: React + Vite
- `server`: Express.js

### Frontend (React + Vite)
- **Dependencies**:
    - `react-router-dom`: For routing (home vs interview room).
    - `@monaco-editor/react`: For the code editor.
    - `socket.io-client`: For real-time communication.
    - `yjs` & `y-monaco` & `y-websocket` (or custom socket.io sync): For CRDT-based collaboration (robust) or simple event broadcasting. *Decision*: I will use `socket.io` broadcasting for simplicity as requested, or `yjs` if robust conflict resolution is preferred. Given "Show real-time updates", simple broadcasting is easier to start, but `yjs` is better. I'll stick to a simple operational transform or broadcast approach for the MVP unless requested otherwise.
- **Components**:
    - `LandingPage`: Button to "Create Interview".
    - `InterviewRoom`: The main workspace.
        - `EditorPanel`: Monaco editor instance.
        - `OutputPanel`: Shows console logs/errors.
        - `Sidebar`: List of connected users.
- **Logic**:
    - **Collaboration**: Emit `code-change` events via Socket.io.
    - **Execution**: Use a `Web Worker` to execute code. Capture `console.log` by overriding it in the worker scope.

### OpenAPI Specs
- Define `openapi.yaml` describing:
    - `POST /api/sessions`: Create a new interview session.
    - `GET /api/sessions/:id`: Get session details (optional, mostly handled via socket).

### Backend (Express.js)
- **Dependencies**:
    - `express`
    - `socket.io`
    - `cors`
- **Logic**:
    - Serve static files (if needed, or just API).
    - **Socket.io**:
        - Handle `join-room`.
        - Broadcast `code-update` to other clients in the room.
        - Handle `cursor-position` updates (bonus).
    - **API**:
        - Generate unique session IDs (UUID).

## Verification Plan

### Automated Tests
- None planned for this MVP phase, focusing on manual verification.

### Manual Verification
1.  **Collaboration**: Open two browser windows with the same link. Type in one, verify update in the other.
2.  **Execution**: Write `console.log("hello")` and run. Verify output in the panel.
3.  **Infinite Loop Protection**: Write a `while(true)` loop and run. Verify the browser doesn't crash (Web Worker termination).
