# app/api/schemas/embed.py
#
# Description:
# This file defines the Pydantic models for the embedding API's request and response.

from pydantic import BaseModel
from typing import List

class EmbedRequest(BaseModel):
    """
    Defines the request body for the /embed endpoint.
    """
    texts: List[str]

class EmbedResponse(BaseModel):
    """
    Defines the response body for the /embed endpoint.
    """
    vectors: List[List[float]]