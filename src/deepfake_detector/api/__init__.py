# src/deepfake_detector/api/__init__.py

"""
API Sub-package for the Deepfake Detection System.

This package contains all the components necessary to expose the deepfake
detection functionality as a web service. It is designed to be a standalone
module that can be run to start a web server, providing endpoints for
image submission and receiving detection results.

The primary entry point is the `create_app` factory function, which
initializes and configures the web application (e.g., a Flask or FastAPI app).

Typical modules within this package would include:
    - app.py: Contains the application factory (`create_app`).
    - routes.py: Defines the API endpoints (e.g., /predict) and their logic.
    - schemas.py: (If using FastAPI) Defines data validation models for requests and responses.
    - dependencies.py: Manages shared dependencies like loading the ML model once.
"""

# The __all__ variable defines the public API of this package.
# When a user does `from deepfake_detector.api import *`, only the
# names listed in `__all__` will be imported.
# We expose the application factory function as the main entry point.
__all__ = ["create_app"]

# To make the application factory easily accessible from the package level,
# we import it here from a submodule (e.g., `app.py`).
# This line assumes the existence of a file `src/deepfake_detector/api/app.py`
# which contains a function `create_app()`.
# This pattern promotes modularity and avoids circular dependencies.
from .app import create_app

```