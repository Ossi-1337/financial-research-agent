from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

SUPPORTED_TYPES = {"object", "string", "number", "integer", "boolean", "array", "null"}


def validate_tool_schema(schema: Mapping[str, Any]) -> tuple[str, ...]:
    errors = _validate_schema(schema, path="schema")
    schema_type = schema.get("type")
    if schema_type != "object":
        errors.append("schema.type must be object")
    return tuple(errors)


def validate_tool_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> tuple[str, ...]:
    if not isinstance(arguments, Mapping):
        return ("arguments must be an object",)
    return tuple(_validate_value(schema, arguments, path="arguments"))


def _validate_schema(schema: object, *, path: str) -> list[str]:
    if not isinstance(schema, Mapping):
        return [f"{path} must be an object"]

    errors: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        if schema_type not in SUPPORTED_TYPES:
            errors.append(f"{path}.type is unsupported: {schema_type}")
    elif isinstance(schema_type, Sequence) and not isinstance(schema_type, str | bytes):
        for index, item in enumerate(schema_type):
            if not isinstance(item, str) or item not in SUPPORTED_TYPES:
                errors.append(f"{path}.type[{index}] is unsupported")
    else:
        errors.append(f"{path}.type is required")

    properties = schema.get("properties", {})
    if properties is not None:
        if not isinstance(properties, Mapping):
            errors.append(f"{path}.properties must be an object")
        else:
            for name, child_schema in properties.items():
                if not isinstance(name, str) or name.strip() == "":
                    errors.append(f"{path}.properties contains an invalid property name")
                    continue
                errors.extend(_validate_schema(child_schema, path=f"{path}.properties.{name}"))

    required = schema.get("required", ())
    if not isinstance(required, Sequence) or isinstance(required, str | bytes):
        errors.append(f"{path}.required must be an array of strings")
    else:
        for index, name in enumerate(required):
            if not isinstance(name, str) or name.strip() == "":
                errors.append(f"{path}.required[{index}] must be a non-empty string")

    additional = schema.get("additionalProperties", True)
    if isinstance(additional, Mapping):
        errors.extend(_validate_schema(additional, path=f"{path}.additionalProperties"))
    elif not isinstance(additional, bool):
        errors.append(f"{path}.additionalProperties must be a boolean or schema object")

    enum_values = schema.get("enum")
    if enum_values is not None and (
        not isinstance(enum_values, Sequence) or isinstance(enum_values, str | bytes)
    ):
        errors.append(f"{path}.enum must be an array")

    items = schema.get("items")
    if items is not None:
        errors.extend(_validate_schema(items, path=f"{path}.items"))

    min_items = schema.get("minItems")
    if min_items is not None and (
        not isinstance(min_items, int) or isinstance(min_items, bool) or min_items < 0
    ):
        errors.append(f"{path}.minItems must be a non-negative integer")
    max_items = schema.get("maxItems")
    if max_items is not None and (
        not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 0
    ):
        errors.append(f"{path}.maxItems must be a non-negative integer")
    if isinstance(min_items, int) and isinstance(max_items, int) and max_items < min_items:
        errors.append(f"{path}.maxItems must be greater than or equal to minItems")

    max_length = schema.get("maxLength")
    if max_length is not None and (
        not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 0
    ):
        errors.append(f"{path}.maxLength must be a non-negative integer")

    return errors


def _validate_value(schema: Mapping[str, Any], value: object, *, path: str) -> list[str]:
    errors: list[str] = []
    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"{path} must be one of {list(enum_values)!r}")

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        allowed_types = (expected_type,)
    elif isinstance(expected_type, Sequence) and not isinstance(expected_type, str | bytes):
        allowed_types = tuple(item for item in expected_type if isinstance(item, str))
    else:
        allowed_types = ()

    if allowed_types and not any(_matches_type(value, item) for item in allowed_types):
        errors.append(f"{path} has invalid type")
        return errors

    if "object" in allowed_types:
        errors.extend(_validate_object(schema, value, path=path))
    elif "array" in allowed_types:
        errors.extend(_validate_array(schema, value, path=path))
    elif "string" in allowed_types and isinstance(value, str):
        max_length = schema.get("maxLength")
        if (
            isinstance(max_length, int)
            and not isinstance(max_length, bool)
            and len(value) > max_length
        ):
            errors.append(f"{path} must contain at most {max_length} characters")
    return errors


def _validate_object(schema: Mapping[str, Any], value: object, *, path: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]

    errors: list[str] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = schema.get("required", ())
    if isinstance(required, Sequence) and not isinstance(required, str | bytes):
        for name in required:
            if isinstance(name, str) and name not in value:
                errors.append(f"{path}.{name} is required")

    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        if not isinstance(key, str):
            errors.append(f"{path} contains a non-string key")
            continue
        child_schema = properties.get(key)
        if isinstance(child_schema, Mapping):
            errors.extend(_validate_value(child_schema, item, path=f"{path}.{key}"))
        elif additional is False:
            errors.append(f"{path}.{key} is not allowed")
        elif isinstance(additional, Mapping):
            errors.extend(_validate_value(additional, item, path=f"{path}.{key}"))
    return errors


def _validate_array(schema: Mapping[str, Any], value: object, *, path: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return [f"{path} must be an array"]
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and not isinstance(min_items, bool) and len(value) < min_items:
        return [f"{path} must contain at least {min_items} items"]
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and not isinstance(max_items, bool) and len(value) > max_items:
        return [f"{path} must contain at most {max_items} items"]
    item_schema = schema.get("items")
    if not isinstance(item_schema, Mapping):
        return []
    errors: list[str] = []
    for index, item in enumerate(value):
        errors.extend(_validate_value(item_schema, item, path=f"{path}[{index}]"))
    return errors


def _matches_type(value: object, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, Real) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, Sequence) and not isinstance(value, str | bytes)
    if expected_type == "null":
        return value is None
    return False
