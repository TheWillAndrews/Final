from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="Grocery Aisle Product Finder",
    description="Upload a product photo and get its aisle + section.",
)

app.include_router(router)

