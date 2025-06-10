# src/deepfake_detector/models/__init__.py

"""
Models Sub-package for the Deepfake Detection System.

This package contains the definitions for the various neural network architectures
used for deepfake detection. It provides a centralized way to access and
instantiate different models.

The `get_model` function serves as a factory to retrieve a model instance
based on its name, which simplifies model selection in the training and
inference pipelines.

To add a new model:
1. Create a new Python file in this directory (e.g., `my_new_model.py`).
2. Define your model class (inheriting from `torch.nn.Module`) in that file.
3. Decorate your class with `@register_model("my_new_model_name")`.
4. Import the new module in this `__init__.py` file (e.g., `from . import my_new_model`).
"""

import logging
from typing import Dict, Type, List

import torch.nn as nn

# A simple registry for available models.
# This dictionary is populated by the @register_model decorator.
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """
    A decorator to register a new model class in the model registry.

    Args:
        name (str): The name to register the model with. Should be unique.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model with name '{name}' is already registered.")
        _MODEL_REGISTRY[name] = cls
        logging.debug(f"Registered model '{name}'")
        return cls
    return decorator


# Import model definitions here to populate the registry.
# This approach ensures that as long as a model file is created and its
# class is decorated with @register_model, it will be available through
# the factory function. The `noqa: F401` comment tells linters to ignore
# the "unused import" warning, as the import's side effect is registration.
try:
    from . import xception  # noqa: F401
    from . import mesonet   # noqa: F401
except ImportError as e:
    # This allows the package to be imported even if some model dependencies are missing,
    # though get_model will fail if a non-existent model is requested.
    logging.warning(
        f"Could not import all model definitions. Some models may not be available. Error: {e}"
    )


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Model factory function.

    Retrieves and instantiates a model by its registered name.

    Args:
        model_name (str): The name of the model to retrieve (e.g., "xception").
                          Must be a key in the model registry.
        **kwargs: Additional keyword arguments to pass to the model's constructor
                  (e.g., `pretrained=True`, `num_classes=1`).

    Returns:
        nn.Module: An instance of the requested model.

    Raises:
        ValueError: If the requested `model_name` is not found in the registry.
    """
    model_name = model_name.lower()
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models are: {list_models()}"
        )

    model_class = _MODEL_REGISTRY[model_name]
    logging.info(f"Initializing model: {model_name} with args: {kwargs}")

    model = model_class(**kwargs)
    return model


def list_models() -> List[str]:
    """
    Returns a list of available model names.

    Returns:
        List[str]: A sorted list of registered model names.
    """
    return sorted(_MODEL_REGISTRY.keys())

```