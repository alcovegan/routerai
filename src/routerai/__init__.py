from .client import RouterAI
from .errors import (
    APIStatusError,
    AuthenticationError,
    InsufficientFundsError,
    ModelNotFoundError,
    NoProviderError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestError,
    RouterAIError,
)
from .registry import Registry
from .schemas import (
    Architecture,
    Capability,
    ChatResult,
    Endpoint,
    Model,
    ModelDetail,
    ModelPricing,
    ProviderSelection,
    ServiceTier,
    ToolCall,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "APIStatusError",
    "Architecture",
    "AuthenticationError",
    "Capability",
    "ChatResult",
    "Endpoint",
    "InsufficientFundsError",
    "Model",
    "ModelDetail",
    "ModelNotFoundError",
    "ModelPricing",
    "NoProviderError",
    "NotFoundError",
    "PermissionDeniedError",
    "ProviderSelection",
    "RateLimitError",
    "Registry",
    "RequestError",
    "RouterAI",
    "RouterAIError",
    "ServiceTier",
    "ToolCall",
    "Usage",
    "__version__",
]
