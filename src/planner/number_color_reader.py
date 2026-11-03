"""Strict Doubao number/background-colour reader.

This module deliberately owns only the image-reading contract.  It does not
plan navigation, choose actions, or perform OCR.  The Ark client is created
lazily so importing the planner never reads credentials or touches the
network.  Tests can inject a client implementing ``responses.create``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .vlm_support import response_text, rgb_data_url


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant")


class DoubaoNumberColorReader:
    """Read a number and its background colour from one current RGB frame."""

    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    DEFAULT_KEY_ENV = "ARK_API_KEY"
    DEFAULT_TIMEOUT_S = 30.0
    DEFAULT_PROMPT = (
        "只识别图片中清晰可见的数字及该数字所在背景的颜色。"
        "不要识别、推断或描述机器人动作、导航或其他内容。"
        "只能输出严格 JSON，字段必须为："
        '{"readable": true, "number": 14, "background_color": "yellow", '
        '"failure_reason": null}。'
        "如果数字或背景颜色任一无法可靠辨认，输出 readable=false，"
        'number=null，background_color=null，并填写 failure_reason。'
    )

    def __init__(
        self,
        model: str,
        *,
        prompt: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_KEY_ENV,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if timeout <= 0 or not math.isfinite(float(timeout)):
            raise ValueError("timeout must be a finite positive number")
        self.model = model
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        # Do not include the secret in an exception or log message.
        if not os.getenv(self.api_key_env):
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
        from volcenginesdkarkruntime import Ark

        self._client = Ark(base_url=self.base_url, api_key=os.environ[self.api_key_env])
        return self._client

    @staticmethod
    def _failure(reason: str) -> dict[str, Any]:
        return {
            "readable": False,
            "number": None,
            "background_color": None,
            "failure_reason": reason,
        }

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        """Validate the model's JSON and return exactly the public schema."""
        candidate = text.strip()
        # Markdown fences are not strict JSON and must not be accepted.
        if not candidate or candidate.startswith("```"):
            return DoubaoNumberColorReader._failure("invalid_json")
        try:
            value = json.loads(
                candidate,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_object,
            )
        except (ValueError, TypeError):
            return DoubaoNumberColorReader._failure("invalid_json")
        if not isinstance(value, dict):
            return DoubaoNumberColorReader._failure("invalid_schema")
        if set(value) - {"readable", "number", "background_color", "failure_reason"}:
            return DoubaoNumberColorReader._failure("invalid_schema")

        readable = value.get("readable")
        if not isinstance(readable, bool):
            return DoubaoNumberColorReader._failure("invalid_readable")
        number = value.get("number")
        color = value.get("background_color")
        reason = value.get("failure_reason")
        if reason is not None and not isinstance(reason, str):
            return DoubaoNumberColorReader._failure("invalid_failure_reason")

        if number is not None:
            # bool is an int subclass but is not a numeric reading.
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                return DoubaoNumberColorReader._failure("invalid_number")
            if isinstance(number, float) and not math.isfinite(number):
                return DoubaoNumberColorReader._failure("invalid_number")
        if color is not None and (not isinstance(color, str) or not color.strip()):
            return DoubaoNumberColorReader._failure("invalid_background_color")

        if readable:
            if number is None:
                return DoubaoNumberColorReader._failure("missing_number")
            if color is None:
                return DoubaoNumberColorReader._failure("missing_background_color")
            if reason is not None:
                return DoubaoNumberColorReader._failure("contradictory_reading")
            return {
                "readable": True,
                "number": number,
                "background_color": color.strip(),
                "failure_reason": None,
            }

        # An unreadable response cannot carry a value into the inspection
        # result, even if the model accidentally included one.
        return {
            "readable": False,
            "number": None,
            "background_color": None,
            "failure_reason": reason.strip() if reason and reason.strip() else "unreadable",
        }

    def read(self, rgb_path: str | Path) -> dict[str, Any]:
        """Submit one current RGB file and return the exact reader schema."""
        try:
            path = Path(rgb_path)
            if not path.is_file():
                return self._failure("image_not_found")
            client = self._ensure_client()
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": rgb_data_url(path)},
                            {"type": "input_text", "text": self.prompt},
                        ],
                    }
                ],
                timeout=self.timeout,
            )
            return self._parse(response_text(response))
        except Exception as exc:
            # Keep API keys, URLs containing credentials, and response bodies
            # out of persisted results.  The exception class is useful for
            # diagnosis without exposing sensitive content.
            return self._failure(f"{type(exc).__name__}")
