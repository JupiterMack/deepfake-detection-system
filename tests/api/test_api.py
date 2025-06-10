# tests/api/test_api.py

"""
Unit tests for the FastAPI API endpoints.

This test suite uses pytest and httpx to test the API layer of the application
in isolation. The core detection pipeline is mocked to ensure that tests are
fast and focus solely on the API's behavior (routing, request/response handling,
error states, etc.).
"""

import io
import json
from unittest.mock import MagicMock

import pytest
import numpy as np
from PIL import Image
from httpx import AsyncClient

# Import the FastAPI app instance from the main application file
from src.deepfake_detector.api.main import app

# Use pytest-asyncio to mark all tests in this file as async
pytestmark = pytest.mark.asyncio


# --- Helper Functions and Fixtures ---

def create_dummy_image_bytes(width: int = 100, height: int = 100) -> bytes:
    """
    Generates the bytes for a simple, black PNG image.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        bytes: The PNG image data as a byte string.
    """
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def test_image_bytes() -> bytes:
    """Pytest fixture providing dummy image bytes for tests."""
    return create_dummy_image_bytes()


@pytest.fixture
def mock_pipeline(mocker) -> MagicMock:
    """
    Mocks the DetectionPipeline and its `load` classmethod.

    This fixture prevents the actual, potentially heavy, model from being loaded
    during tests. It patches `DetectionPipeline.load` to return a mock instance,
    which can then be configured within individual tests.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        MagicMock: The mock instance of the DetectionPipeline.
    """
    mock_instance = MagicMock()
    # The `load` method is a classmethod that returns a pipeline instance.
    # We patch it to return our mock instance instead.
    mocker.patch(
        "src.deepfake_detector.core.pipeline.DetectionPipeline.load",
        return_value=mock_instance
    )
    return mock_instance


@pytest.fixture
async def client(mock_pipeline: MagicMock) -> AsyncClient:
    """
    Provides an httpx AsyncClient for the FastAPI app.

    This fixture depends on `mock_pipeline` to ensure the pipeline is mocked
    before the app and its startup events are initialized by the client.

    Args:
        mock_pipeline: The mocked pipeline fixture (ensures it runs first).

    Yields:
        AsyncClient: An asynchronous test client.
    """
    # The app's startup event calls `DetectionPipeline.load`, which is now mocked.
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# --- Test Cases ---

async def test_health_check(client: AsyncClient):
    """
    Tests the root health check endpoint (`/api/v1/`).
    """
    response = await client.get("/api/v1/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Deepfake Detection API is running."}


async def test_detect_fake_image_no_explanation(
    client: AsyncClient, mock_pipeline: MagicMock, test_image_bytes: bytes
):
    """
    Tests the /detect/ endpoint for a FAKE image without requesting an explanation map.
    It should return a standard JSON response.
    """
    # 1. Configure the mock pipeline's return value for this specific test
    mock_prediction_result = {
        "label": "FAKE",
        "confidence": 0.9876,
        "real_prob": 0.0124,
        "fake_prob": 0.9876,
    }
    mock_pipeline.predict.return_value = (mock_prediction_result, None)

    # 2. Prepare the request
    files = {"image_file": ("test.png", test_image_bytes, "image/png")}
    
    # 3. Make the API call
    response = await client.post("/api/v1/detect/", files=files)

    # 4. Assert the response
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    
    data = response.json()
    assert data["label"] == "FAKE"
    assert "confidence" in data
    assert "real_prob" in data
    assert "fake_prob" in data
    assert data["confidence"] == pytest.approx(0.9876)

    # 5. Verify the mocked method was called correctly
    mock_pipeline.predict.assert_called_once()
    call_kwargs = mock_pipeline.predict.call_args.kwargs
    assert call_kwargs['image_bytes'] == test_image_bytes
    assert call_kwargs['generate_explanation'] is False


async def test_detect_real_image_with_explanation(
    client: AsyncClient, mock_pipeline: MagicMock, test_image_bytes: bytes
):
    """
    Tests the /detect/ endpoint for a REAL image with a request for an explanation map.
    It should return a multipart/form-data response.
    """
    # 1. Configure the mock pipeline's return value
    mock_prediction_result = {
        "label": "REAL",
        "confidence": 0.9543,
        "real_prob": 0.9543,
        "fake_prob": 0.0457,
    }
    mock_explanation_bytes = create_dummy_image_bytes(50, 50)
    mock_pipeline.predict.return_value = (mock_prediction_result, mock_explanation_bytes)

    # 2. Prepare and make the request
    files = {"image_file": ("test.png", test_image_bytes, "image/png")}
    params = {"generate_explanation": "true"}
    response = await client.post("/api/v1/detect/", files=files, params=params)

    # 3. Assert the multipart response
    assert response.status_code == 200
    content_type_header = response.headers['content-type']
    assert content_type_header.startswith("multipart/form-data")

    # 4. Parse the multipart body to verify its contents
    boundary = content_type_header.split("boundary=")[1].encode('utf-8')
    body = await response.aread()
    parts = body.split(b'--' + boundary)
    
    # Expected parts: empty, json, image, final empty with '--'
    assert len(parts) == 4

    # --- Verify JSON part ---
    json_part = parts[1]
    assert b'Content-Disposition: form-data; name="prediction"' in json_part
    assert b'Content-Type: application/json' in json_part
    json_str = json_part.split(b'\r\n\r\n')[1].strip()
    prediction_data = json.loads(json_str)
    assert prediction_data["label"] == "REAL"
    assert prediction_data["confidence"] == pytest.approx(0.9543)

    # --- Verify Image part ---
    image_part = parts[2]
    assert b'Content-Disposition: form-data; name="explanation_map"; filename="explanation.png"' in image_part
    assert b'Content-Type: image/png' in image_part
    image_data = image_part.split(b'\r\n\r\n')[1].strip()
    assert image_data == mock_explanation_bytes

    # 5. Verify the mocked method was called correctly
    mock_pipeline.predict.assert_called_once()
    call_kwargs = mock_pipeline.predict.call_args.kwargs
    assert call_kwargs['image_bytes'] == test_image_bytes
    assert call_kwargs['generate_explanation'] is True


async def test_detect_no_file_uploaded(client: AsyncClient):
    """
    Tests the /detect/ endpoint when no file is provided in the request.
    FastAPI should return a 422 Unprocessable Entity error.
    """
    response = await client.post("/api/v1/detect/")
    
    assert response.status_code == 422
    data = response.json()