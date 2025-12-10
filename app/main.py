import uvicorn
from .api import app as fastapi_app  # import the FastAPI instance from app.api

# This is what uvicorn looks for when you run `uvicorn app.main:app`
app = fastapi_app


def main() -> None:
    uvicorn.run(
        "app.main:app",  # now point to this module, not app.api
        host="0.0.0.0",
        port=8000,
        reload=False,  # you can set True if you want reload when running main()
    )


if __name__ == "__main__":
    main()

