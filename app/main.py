from dotenv import load_dotenv
import os
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from app.rate_limiter import rate_limit



load_dotenv()

from fastapi import FastAPI, Depends, HTTPException
from app.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionResponse,
    ErrorResponse
)
api_key = os.getenv("API_KEY")
from app.security import validate_api_key
from app.inference import predict_voice

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse}
    }
)
def voice_detection_endpoint(
    request: VoiceDetectionRequest,
    api_key: str = Depends(validate_api_key),
    _: None = Depends(lambda: rate_limit(api_key))
):

        
    try:
        classification, confidence, explanation = predict_voice(
            request.audioBase64
        )

        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 2),
            explaination=explanation
        )

    except ValueError as e:
       return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": str(e)
            }
        )

                
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )

