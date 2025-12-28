# core/container.py
#
# DEPRECATED: This file has been moved to infrastructure/container.py
# 
# This file is kept for backward compatibility only.
# Please update your imports to use:
#   from infrastructure.container import get_container, get_search_service
#
# The DI Container is part of the infrastructure layer (composition root)
# and has been moved to the appropriate location.

import warnings
from infrastructure.container import (
    DIContainer,
    get_container,
    get_search_service,
    reset_container
)

# Issue deprecation warning
warnings.warn(
    "Importing from core.container is deprecated. "
    "Please use 'from infrastructure.container import get_container, get_search_service' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['DIContainer', 'get_container', 'get_search_service', 'reset_container']
