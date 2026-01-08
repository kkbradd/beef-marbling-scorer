"""
FastAPI application for CowinBMS inference API.
"""
import os
import sys
from pathlib import Path
from typing import Optional, List
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    def limiter_limit(limit: str):
        def decorator(func):
            return func
        return decorator
    Limiter = None

import uvicorn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference_engine import get_engine
from src.utils.config import load_config
from src.utils.validators import ValidationError
from src.utils.batch_output import export_batch_results
from src.utils.logging_config import get_logger

logger = get_logger("api")

app = FastAPI(
    title="CowinBMS API",
    description="AI-powered beef quality assessment API",
    version="1.0.0"
)

if RATE_LIMITING_AVAILABLE and Limiter:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    limiter = None
    logger.warning("Rate limiting not available. Install slowapi for rate limiting.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()
engine = get_engine(config)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "CowinBMS API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single image prediction",
            "/batch": "POST - Batch image prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine.load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    apply_segmentation: bool = Form(True),
    save_visualization: bool = Form(False)
):
    """
    Predict beef quality from single image.
    
    Rate limit: 10 requests per minute (if slowapi is installed).
    
    Args:
        request: FastAPI request object (for rate limiting)
        file: Image file to process
        apply_segmentation: Whether to apply segmentation
        save_visualization: Whether to save visualization
    
    Returns:
        Prediction results
    """
    if RATE_LIMITING_AVAILABLE and limiter:
        @limiter.limit("10/minute")
        async def _limited_predict():
            pass
        await _limited_predict()
    
    allowed_extensions = config.get("api.allowed_extensions", [".jpg", ".jpeg", ".png"])
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    max_size = config.get("api.max_file_size", 10485760)
    contents = await file.read()
    
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(contents)} bytes. Maximum: {max_size} bytes"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(contents)
        tmp_path = tmp_file.name
    
    try:
        result = engine.predict(
            tmp_path,
            apply_segmentation=apply_segmentation,
            save_visualization=save_visualization,
            log_prediction=True
        )
        
        os.unlink(tmp_path)
        
        return result
    
    except ValidationError as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        os.unlink(tmp_path)
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch")
async def batch_predict(
    request: Request,
    files: List[UploadFile] = File(...),
    apply_segmentation: bool = Form(True),
    export_format: str = Form("json")
):
    """
    Batch prediction for multiple images.
    
    Rate limit: 5 batch requests per minute (if slowapi is installed).
    
    Args:
        request: FastAPI request object (for rate limiting)
        files: List of image files
        apply_segmentation: Whether to apply segmentation
        export_format: Export format ('json', 'csv', 'excel', or 'all')
    
    Returns:
        Batch prediction results
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    allowed_extensions = config.get("api.allowed_extensions", [".jpg", ".jpeg", ".png"])
    max_size = config.get("api.max_file_size", 10485760)
    
    tmp_paths = []
    predictions = []
    errors = []
    
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            errors.append({
                "filename": file.filename,
                "error": f"Invalid file type: {file_ext}"
            })
            continue
        
        contents = await file.read()
        
        if len(contents) > max_size:
            errors.append({
                "filename": file.filename,
                "error": f"File too large: {len(contents)} bytes"
            })
            continue
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
            tmp_paths.append(tmp_path)
        
        try:
            result = engine.predict(
                tmp_path,
                apply_segmentation=apply_segmentation,
                save_visualization=False,
                log_prediction=True
            )
            predictions.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
            predictions.append(None)
    
    for tmp_path in tmp_paths:
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    export_files = {}
    if export_format in ['csv', 'excel', 'json', 'all']:
        output_dir = config.get_absolute_path("outputs/exports")
        formats = ['csv', 'excel', 'json'] if export_format == 'all' else [export_format]
        
        export_files = export_batch_results(
            predictions,
            str(output_dir),
            formats=formats
        )
    
    return {
        "summary": {
            "total_files": len(files),
            "successful": len([p for p in predictions if p is not None]),
            "errors": len(errors)
        },
        "predictions": [p for p in predictions if p is not None],
        "errors": errors,
        "export_files": export_files
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    run_server(host=host, port=port)

