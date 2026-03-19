from fastapi import FastAPI
from src.router import router as process_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(process_router)

origins = [
    "http://localhost:5173",
    "https://webresearchai.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
