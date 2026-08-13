from .client import RouterAI
from .errors import (
    APIStatusError,
    AuthenticationError,
    ConfigurationError,
    DeadlineExceededError,
    InsufficientFundsError,
    ModelNotFoundError,
    NoProviderError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestError,
    RouterAIError,
    StreamInterruptedError,
    VideoGenerationError,
    WebhookVerificationError,
)
from .registry import Registry
from .resources.chat import AudioDelta, StreamAccumulator
from .resources.videos import FrameImage, ImageReference
from .schemas import (
    Architecture,
    Capability,
    ChatResult,
    Endpoint,
    KeyInfo,
    MemberCreation,
    MessagesResult,
    Model,
    ModelDetail,
    ModelPricing,
    ProviderSelection,
    ResponsesResult,
    ServiceTier,
    TeamInvitation,
    TeamMember,
    ToolCall,
    Usage,
)

__version__: str
try:
    from importlib.metadata import version

    __version__ = version("routerai")
except Exception:  # pragma: no cover - source checkout without installed metadata
    __version__ = "0.1.0"

__all__ = [
    "APIStatusError",
    "Architecture",
    "AudioDelta",
    "AuthenticationError",
    "Capability",
    "ChatResult",
    "ConfigurationError",
    "DeadlineExceededError",
    "Endpoint",
    "FrameImage",
    "ImageReference",
    "InsufficientFundsError",
    "KeyInfo",
    "MemberCreation",
    "MessagesResult",
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
    "ResponsesResult",
    "RouterAI",
    "RouterAIError",
    "ServiceTier",
    "StreamAccumulator",
    "StreamInterruptedError",
    "TeamInvitation",
    "TeamMember",
    "ToolCall",
    "Usage",
    "VideoGenerationError",
    "WebhookVerificationError",
    "__version__",
]
