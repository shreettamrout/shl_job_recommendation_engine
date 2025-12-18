import re
from typing import Optional


def extract_slug(url: str) -> Optional[str]:
    """
    Extract canonical assessment slug from SHL assessment URL.

    Example:
    https://www.shl.com/products/product-catalog/view/core-java-entry-level/
    → core-java-entry-level
    """
    if not isinstance(url, str):
        return None

    url = url.lower().strip()
    match = re.search(r"/view/([^/]+)/?", url)

    return match.group(1) if match else None


def normalize_test_types(test_type: str):
    """
    Normalize test_type string into a list.

    Example:
    'A, B, P' → ['A', 'B', 'P']
    """
    if not isinstance(test_type, str):
        return []

    return [t.strip() for t in test_type.split(",") if t.strip()]


def safe_bool(value):
    """
    Convert different truthy/falsey representations to bool.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1"}

    if isinstance(value, (int, float)):
        return value == 1

    return False

