def truncate(text: str, max_len: int = 200) -> str:
    """ Truncates a text to a maximum length, adding ellipsis if truncated. """
    return text if len(text) <= max_len else text[:max_len] + "..."

def enforce_no_additional_properties(schema: dict) -> dict:
    """ Recursively enforce "additionalProperties": False in a JSON schema dict.

    Args:
        schema (dict):  The JSON schema to modify.

    Returns:
        dict: The modified JSON schema.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)

        for key, value in schema.items():
            enforce_no_additional_properties(value)

    elif isinstance(schema, list):
        for item in schema:
            enforce_no_additional_properties(item)

    return schema
