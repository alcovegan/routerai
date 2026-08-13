from __future__ import annotations

import base64
import binascii
import ipaddress
import re
import urllib.parse

DEFAULT_MAX_INLINE_IMAGE_BYTES = 50 * 1024 * 1024

_IMAGE_DATA_URI = re.compile(r"data:image/[a-z0-9.+-]+;base64,([A-Za-z0-9+/]+={0,2})")


def validate_public_https_url(value: str, *, field: str = "url") -> str:
    """Validate an absolute public HTTPS URL without resolving DNS.

    Literal private, loopback, link-local and otherwise non-public IPs are
    rejected. Hostnames are checked structurally; DNS resolution and rebinding
    protection remain the responsibility of the system that performs the
    outbound request.
    """
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{field} must not contain whitespace or control characters")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid URL") from exc
    if parsed.scheme != "https" or not host:
        raise ValueError(f"{field} must be an absolute https url")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} with embedded credentials is not allowed")
    if parsed.fragment:
        raise ValueError(f"{field} with a fragment is not allowed")
    if port == 0:
        raise ValueError(f"{field} has an invalid port")

    normalized_host = host.lower()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise ValueError(f"{field} host is not public: {normalized_host!r}")
    try:
        ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        return value
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise ValueError(f"{field} host is not a public address: {normalized_host!r}")
    return value


def validate_image_source(
    value: str, *, max_data_bytes: int = DEFAULT_MAX_INLINE_IMAGE_BYTES
) -> str:
    """Validate a public HTTPS image URL or a bounded base64 image data URI."""
    if not value.startswith("data:"):
        return validate_public_https_url(value, field="image url")

    match = _IMAGE_DATA_URI.fullmatch(value)
    if not match:
        raise ValueError("image data uri must look like data:image/<type>;base64,<payload>")
    payload = match.group(1)
    padding = len(payload) - len(payload.rstrip("="))
    decoded_size = (len(payload) * 3) // 4 - padding
    if decoded_size > max_data_bytes:
        raise ValueError(f"image data uri exceeds the {max_data_bytes} byte limit")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data uri carries invalid base64 payload") from exc
    if not decoded:
        raise ValueError("image data uri payload is empty")
    return value
