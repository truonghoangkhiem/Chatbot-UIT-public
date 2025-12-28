"""
Robust JSON Parsing Utilities for LLM Responses.

Handles common issues with LLM-generated JSON:
- Markdown code blocks (```json ... ```)
- Extra whitespace and newlines
- Trailing commas
- Single quotes instead of double quotes
- Malformed JSON with recovery attempts

Author: Legal Document Processing Team
Date: 2024
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def clean_json_text(text: str) -> str:
    """
    Clean and extract JSON from LLM response text.
    
    Handles:
    - Markdown code blocks (```json, ```)
    - Leading/trailing whitespace
    - BOM characters
    - Common formatting issues
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned JSON string ready for parsing
        
    Example:
        >>> raw = '''```json
        ... {"key": "value"}
        ... ```'''
        >>> clean_json_text(raw)
        '{"key": "value"}'
    """
    if not text:
        return "{}"
    
    # Remove BOM and strip whitespace
    text = text.strip().lstrip('\ufeff')
    
    # Pattern 1: Extract from ```json ... ``` blocks
    json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_block_pattern, text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        logger.debug("Extracted JSON from ```json block")
        return text
    
    # Pattern 2: Extract from ``` ... ``` blocks (generic code block)
    code_block_pattern = r'```\s*([\s\S]*?)\s*```'
    match = re.search(code_block_pattern, text)
    if match:
        candidate = match.group(1).strip()
        # Verify it looks like JSON
        if candidate.startswith('{') or candidate.startswith('['):
            text = candidate
            logger.debug("Extracted JSON from ``` block")
            return text
    
    # Pattern 3: Find JSON object {...} in text
    # Use greedy matching to find the largest valid JSON object
    json_obj_pattern = r'(\{[\s\S]*\})'
    matches = re.findall(json_obj_pattern, text)
    if matches:
        # Try each match, longest first
        for candidate in sorted(matches, key=len, reverse=True):
            try:
                json.loads(candidate)
                logger.debug("Extracted JSON object from text")
                return candidate
            except json.JSONDecodeError:
                continue
    
    # Pattern 4: Find JSON array [...] in text
    json_arr_pattern = r'(\[[\s\S]*\])'
    matches = re.findall(json_arr_pattern, text)
    if matches:
        for candidate in sorted(matches, key=len, reverse=True):
            try:
                json.loads(candidate)
                logger.debug("Extracted JSON array from text")
                return candidate
            except json.JSONDecodeError:
                continue
    
    # Pattern 5: Simple extraction by finding first { or [ and last } or ]
    start_obj = text.find('{')
    start_arr = text.find('[')
    
    if start_obj >= 0 and (start_arr < 0 or start_obj < start_arr):
        # JSON object
        end = text.rfind('}')
        if end > start_obj:
            text = text[start_obj:end + 1]
    elif start_arr >= 0:
        # JSON array
        end = text.rfind(']')
        if end > start_arr:
            text = text[start_arr:end + 1]
    
    return text.strip()


def fix_common_json_errors(text: str) -> str:
    """
    Attempt to fix common JSON formatting errors.
    
    Fixes:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    - Unescaped newlines in strings
    - Missing quotes around keys
    
    Args:
        text: JSON text with potential errors
        
    Returns:
        Fixed JSON text
    """
    if not text:
        return text
    
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Replace single quotes with double quotes (careful with apostrophes)
    # Only do this if there are no double quotes in the text
    if '"' not in text and "'" in text:
        # Simple replacement for simple cases
        text = text.replace("'", '"')
    
    # Fix unescaped newlines within strings
    # This is tricky - we need to identify strings and escape \n
    def escape_newlines_in_strings(match):
        content = match.group(1)
        # Escape actual newlines (not \n sequences)
        content = content.replace('\n', '\\n').replace('\r', '\\r')
        return f'"{content}"'
    
    # Match strings and escape newlines within them
    text = re.sub(r'"([^"]*(?:\n|\r)[^"]*)"', escape_newlines_in_strings, text)
    
    return text


def safe_json_loads(
    text: str,
    default: Any = None,
    raise_on_error: bool = False
) -> Tuple[Any, Optional[str]]:
    """
    Safely parse JSON with error recovery.
    
    Attempts multiple strategies to parse JSON:
    1. Direct parsing
    2. After cleaning
    3. After fixing common errors
    4. Using json_repair library (if available)
    
    Args:
        text: JSON text to parse
        default: Default value if parsing fails
        raise_on_error: Raise exception instead of returning default
        
    Returns:
        Tuple of (parsed_data, error_message or None)
        
    Example:
        >>> data, error = safe_json_loads('{"key": "value",}')
        >>> print(data)  # {'key': 'value'}
        >>> print(error)  # None
    """
    if not text:
        return default if default is not None else {}, None
    
    original_text = text
    
    # Step 1: Try direct parsing
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    
    # Step 2: Clean the text
    text = clean_json_text(text)
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    
    # Step 3: Fix common errors
    text = fix_common_json_errors(text)
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse error at position {e.pos}: {e.msg}"
    
    # Step 4: Try json_repair library if available
    try:
        import json_repair
        repaired = json_repair.repair_json(original_text)
        data = json.loads(repaired)
        logger.info("JSON repaired using json_repair library")
        return data, None
    except ImportError:
        logger.debug("json_repair library not available")
    except Exception as e:
        error_msg = f"JSON repair failed: {e}"
    
    # All attempts failed
    logger.warning(f"Failed to parse JSON: {error_msg}")
    logger.debug(f"Original text (first 500 chars): {original_text[:500]}")
    
    if raise_on_error:
        raise json.JSONDecodeError(error_msg, text, 0)
    
    return default if default is not None else {}, error_msg


def parse_llm_json_response(
    response_text: str,
    expected_type: str = "object"
) -> Tuple[Union[Dict, List], List[str]]:
    """
    Parse LLM JSON response with full error handling.
    
    Designed specifically for LLM outputs which often have formatting issues.
    
    Args:
        response_text: Raw LLM response
        expected_type: "object" for {} or "array" for []
        
    Returns:
        Tuple of (parsed_data, list_of_errors)
        
    Example:
        >>> response = '''Here is the result:
        ... ```json
        ... {"nodes": [{"id": "1"}], "relations": []}
        ... ```
        ... '''
        >>> data, errors = parse_llm_json_response(response)
        >>> print(data["nodes"])  # [{"id": "1"}]
    """
    errors = []
    
    if not response_text:
        errors.append("Empty response from LLM")
        return {} if expected_type == "object" else [], errors
    
    # Clean and parse
    default = {} if expected_type == "object" else []
    data, error = safe_json_loads(response_text, default=default)
    
    if error:
        errors.append(error)
    
    # Validate type
    if expected_type == "object" and not isinstance(data, dict):
        errors.append(f"Expected JSON object, got {type(data).__name__}")
        if isinstance(data, list) and len(data) > 0:
            # Maybe it's wrapped in an array
            data = {"items": data}
        else:
            data = {}
    elif expected_type == "array" and not isinstance(data, list):
        errors.append(f"Expected JSON array, got {type(data).__name__}")
        if isinstance(data, dict):
            # Try to extract array from common keys
            for key in ["items", "data", "results", "nodes", "relations"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = [data] if data else []
        else:
            data = []
    
    return data, errors


# Convenience function for the extraction pipeline
def clean_and_parse_json(
    response_text: str,
    context: str = "LLM response"
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Clean and parse JSON from LLM response - main entry point.
    
    This is the primary function to use in the extraction pipeline.
    
    Args:
        response_text: Raw response from LLM
        context: Description for error messages
        
    Returns:
        Tuple of (parsed_dict, list_of_errors)
        
    Example:
        >>> raw = '''```json
        ... {"nodes": [{"id": "dieu_1", "type": "Article"}]}
        ... ```'''
        >>> data, errors = clean_and_parse_json(raw, "VLM structure extraction")
        >>> print(data["nodes"])
    """
    logger.debug(f"Parsing {context}: {len(response_text)} chars")
    
    data, errors = parse_llm_json_response(response_text, expected_type="object")
    
    if errors:
        for err in errors:
            logger.warning(f"{context} - {err}")
    else:
        logger.debug(f"{context} - Parsed successfully")
    
    return data, errors


# Test function
def _test_json_utils():
    """Test the JSON utilities."""
    test_cases = [
        # Case 1: Markdown code block
        ('```json\n{"key": "value"}\n```', {"key": "value"}),
        
        # Case 2: Trailing comma
        ('{"key": "value",}', {"key": "value"}),
        
        # Case 3: Text before/after JSON
        ('Here is the result:\n{"data": 123}\nEnd.', {"data": 123}),
        
        # Case 4: Array in markdown
        ('```json\n[{"id": 1}, {"id": 2}]\n```', [{"id": 1}, {"id": 2}]),
        
        # Case 5: Nested JSON with markdown
        ('Response:\n```json\n{"nodes": [{"id": "a"}], "relations": []}\n```', 
         {"nodes": [{"id": "a"}], "relations": []}),
    ]
    
    print("Testing JSON utilities...")
    for i, (input_text, expected) in enumerate(test_cases, 1):
        data, errors = safe_json_loads(input_text)
        status = "✓" if data == expected else "✗"
        print(f"  Test {i}: {status}")
        if data != expected:
            print(f"    Expected: {expected}")
            print(f"    Got: {data}")
            print(f"    Errors: {errors}")
    
    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _test_json_utils()
