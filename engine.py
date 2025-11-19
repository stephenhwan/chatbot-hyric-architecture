from pathlib import Path
import requests

# ===== Configuration =====
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
LM_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_MODEL = "llama-3.2-1b-instruct"
API_KEY_FILE = ".key"

# Backend settings
BACKENDS = {
    "gemini": {"timeout": 60, "max_tokens": 2048},
    "local": {"timeout": 120, "max_tokens": 2048},
}
CURRENT_BACKEND = "auto"  # "auto", "gemini", or "local"


# ===== Utilities =====
def _load_api_key() -> str:
    """Load Gemini API key from file."""
    key_path = Path(__file__).resolve().parent / API_KEY_FILE
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Gemini API key file is empty")
    return key


def _make_request(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    """Generic HTTP POST request with error handling."""
    res = requests.post(url, headers=headers, json=payload, timeout=timeout)
    res.raise_for_status()
    return res.json()


# ===== Backend Implementations =====
class GeminiBackend:
    """Gemini API backend."""

    @staticmethod
    def format_history(history: list[dict]) -> list[dict]:
        """Convert history to Gemini format (skip system messages)."""
        contents = []
        for msg in history:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if not text or role == "system":
                continue
            api_role = "model" if role == "assistant" else "user"
            contents.append({"role": api_role, "parts": [{"text": text}]})
        return contents

    @staticmethod
    def extract_reply(data: dict) -> str:
        """Extract text from Gemini response."""
        candidates = data.get("candidates", [])
        if not candidates:
            feedback = data.get("promptFeedback")
            raise RuntimeError(f"No candidates returned: {feedback}")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        if not parts:
            finish_reason = candidates[0].get("finishReason")
            safety = candidates[0].get("safetyRatings", [])
            raise RuntimeError(f"Empty parts. Reason: {finish_reason}, Safety: {safety}")

        reply = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if not reply:
            raise RuntimeError(f"Empty reply after joining. Parts: {parts}")
        return reply

    @classmethod
    def chat(cls, msg: str, history: list[dict]) -> str:
        """Call Gemini API."""
        config = BACKENDS["gemini"]
        tmp_history = history + [{"role": "user", "content": msg}]

        payload = {
            "contents": cls.format_history(tmp_history),
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": config["max_tokens"],
            },
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": _load_api_key(),
        }

        data = _make_request(GEMINI_URL, payload, headers, config["timeout"])
        return cls.extract_reply(data)


class LocalBackend:

    @staticmethod
    def format_history(history: list[dict], new_msg: str) -> list[dict]:
        """Convert history to OpenAI format."""
        messages = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": new_msg})
        return messages

    @staticmethod
    def extract_reply(data: dict) -> str:
        """Extract text from OpenAI-style response."""
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned")

        reply = choices[0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("Empty reply")
        return reply

    @classmethod
    def chat(cls, msg: str, history: list[dict]) -> str:
        """Call local LM Studio API."""
        config = BACKENDS["local"]

        payload = {
            "model": LM_MODEL,
            "messages": cls.format_history(history, msg),
            "temperature": 0.7,
            "max_tokens": config["max_tokens"],
        }

        headers = {"Content-Type": "application/json"}

        data = _make_request(LM_URL, payload, headers, config["timeout"])
        return cls.extract_reply(data)


# ===== Backend Router =====
class BackendRouter:
    """Route requests to appropriate backend with fallback."""

    BACKENDS_MAP = {
        "gemini": GeminiBackend,
        "local": LocalBackend,
    }

    @classmethod
    def _try_backend(cls, backend_name: str, msg: str, history: list[dict]) -> str:
        """Try calling a specific backend."""
        backend = cls.BACKENDS_MAP[backend_name]
        try:
            return backend.chat(msg, history)
        except Exception as e:
            raise RuntimeError(f"[{backend_name}] {e}")

    @classmethod
    def route(cls, msg: str, history: list[dict], backend: str = "auto") -> str:
        """Route request to backend(s) based on strategy."""
        if backend == "gemini":
            return cls._try_backend("gemini", msg, history)

        if backend == "local":
            return cls._try_backend("local", msg, history)

        # Auto mode: try Gemini first, fallback to local
        try:
            return cls._try_backend("gemini", msg, history)
        except Exception as e_gemini:
            print(f"[Error] Gemini failed, trying local llama-3.2-1")
            try:
                return cls._try_backend("local", msg, history)
            except Exception as e_local:
                return f"Both backends failed.\nGemini: {e_gemini}\nLocal: {e_local}"


# ===== Public API =====
def chat(msg: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:

    if history is None:
        history = [{"role": "system", "content": "You are a helpful hotel + ML assistant + answer short."}]

    reply = BackendRouter.route(msg, history, backend=CURRENT_BACKEND)

    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": reply})

    return reply, history