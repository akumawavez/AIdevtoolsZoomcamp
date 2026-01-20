from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AI Aptitude Test Platform API")

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthCheck(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthCheck)
async def root():
    return {"status": "ok", "message": "Backend is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
