# src/deepfake_detector/api/main.py

"""
Main entry point for the Deepfake Detection System FastAPI application.

This script initializes the FastAPI application, sets up event handlers for
startup and shutdown (e.g., loading the model), includes the API routers,
and provides a main execution block to run the server using Uvicorn.
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import application-specific modules
# Assuming a config module for settings management and an inference module for model loading
from .. import config
from ..inference import load_model_and_dependencies
from .endpoints import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
# The app metadata will be displayed in the auto-generated docs (e.g., /docs)
app = FastAPI(
    title=config.settings.PROJECT_NAME,
    description="A robust system that uses AI to detect manipulated images (deepfakes). "
                "This API provides endpoints to analyze images and identify potential manipulations, "
                "including explainable AI to highlight suspicious regions.",
    version="1.0.0",
    contact={
        "name": "Project Maintainer",
        "url": "https://github.com/your-username/deepfake-detection-system",  # Placeholder URL
        "email": "maintainer@example.com",  # Placeholder email
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# --- Application State ---
# Use app.state to store objects that should be available across the application's lifespan,
# such as the machine learning model. This avoids reloading the model on every request.
# These will be populated by the 'startup' event handler.
app.state.model = None
app.state.transformer = None
app.state.device = None


# --- Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.

    This function is executed when the application starts. It's responsible for
    loading the machine learning model and other necessary resources into the
    application's state, making them available for request handlers.
    """
    logger.info("Application startup: Loading resources...")
    try:
        # Load the model, transformer, and device using the utility function
        # The function is expected to read the model path from the settings.
        model, transformer, device = load_model_and_dependencies(model_path=config.settings.MODEL_PATH)

        # Store the loaded objects in the application state
        app.state.model = model
        app.state.transformer = transformer
        app.state.device = device

        logger.info(f"Model '{config.settings.MODEL_PATH}' and transformer loaded successfully.")
        logger.info(f"Inference will run on device: {app.state.device}")
    except Exception as e:
        logger.error(f"FATAL: Error during application startup: {e}", exc_info=True)
        # Depending on the desired behavior, you might want to exit the application
        # if the model fails to load. For now, we log the error and let it continue,
        # but endpoints will likely fail with a 500 error.
        # To force exit:
        # import sys
        # sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Event handler for application shutdown.

    This function is executed when the application is shutting down. It's used
    for cleanup tasks, such as releasing resources.
    """
    logger.info("Application shutdown: Cleaning up resources...")
    # Clear the state to release memory
    app.state.model = None
    app.state.transformer = None
    app.state.device = None
    logger.info("Resources cleaned up successfully.")


# --- Middleware ---

# Add CORS middleware to allow cross-origin requests, e.g., from a web frontend.
# You should restrict the origins to your specific frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- API Routers ---

# Include the router from endpoints.py
# All routes defined in that router will be prefixed as defined in the config
app.include_router(api_router, prefix=config.settings.API_V1_STR)


# --- Root Endpoint ---

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint providing a simple status message and a link to the docs.
    Useful for health checks and service discovery.
    """
    return {
        "status": "ok",
        "message": f"Welcome to the {config.settings.PROJECT_NAME}. Visit /docs for the API documentation."
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    # This block allows running the application directly using `python -m src.deepfake_detector.api.main`
    # Uvicorn is a lightning-fast ASGI server implementation.
    # It's configured here to run the FastAPI app instance.
    # The host and port are sourced from the settings module.
    # `reload=True` is useful for development, as it automatically restarts the server on code changes.
    # For production, this should be False and managed by a process manager like Gunicorn.

    # To run from the project root:
    # PYTHONPATH=. uvicorn src.deepfake_detector.api.main:app --host 0.0.0.0 --port 8000 --reload
    
    uvicorn.run(
        "src.deepfake_detector.api.main:app",
        host=config.settings.SERVER_HOST,
        port=config.settings.SERVER_PORT,
        reload=config.settings.SERVER_RELOAD,
        log_level="info"
    )
```