"""Template profiles: which section builder fits the server's chat template."""

PROFILES = ("chatml_qwen", "generic")


def detect_profile(chat_template):
    """ChatML with Qwen-style tool markup -> "chatml_qwen"; anything else -> "generic"."""
    t = chat_template or ""
    if "<|im_start|>" in t and "<tools>" in t and "<tool_call>" in t:
        return "chatml_qwen"
    return "generic"
