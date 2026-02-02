import os
from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse
API_KEY = os.getenv("API_KEY")

def validate_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )

