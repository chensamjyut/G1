from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from planner.number_color_reader import DoubaoNumberColorReader


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _Response:
    def __init__(self, text: str):
        self.output_text = text


class _Responses:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.text)


class _Client:
    def __init__(self, text: str):
        self.responses = _Responses(text)


class _RaisingResponses:
    def create(self, **kwargs):
        raise TimeoutError("private response details must not escape")


class _RaisingClient:
    responses = _RaisingResponses()


class NumberColorReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "current.png"
        self.image.write_bytes(PNG)

    def read(self, text: str):
        client = _Client(text)
        reader = DoubaoNumberColorReader("model-test", client=client, timeout=7.5)
        result = reader.read(self.image)
        return result, client

    def test_success_has_exact_schema_and_request_contract(self):
        result, client = self.read(
            '{"readable":true,"number":14,"background_color":"yellow","failure_reason":null}'
        )
        self.assertEqual(result, {
            "readable": True, "number": 14,
            "background_color": "yellow", "failure_reason": None,
        })
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "model-test")
        self.assertEqual(call["timeout"], 7.5)
        self.assertEqual(call["input"][0]["content"][0]["type"], "input_image")

    def test_unreadable_is_normalized(self):
        result, _ = self.read('{"readable":false,"number":null,"background_color":null,"failure_reason":"blurred"}')
        self.assertEqual(result["readable"], False)
        self.assertIsNone(result["number"])
        self.assertIsNone(result["background_color"])
        self.assertEqual(result["failure_reason"], "blurred")

    def test_missing_color_is_not_success(self):
        result, _ = self.read('{"readable":true,"number":14,"background_color":null,"failure_reason":null}')
        self.assertEqual(result, {
            "readable": False, "number": None,
            "background_color": None, "failure_reason": "missing_background_color",
        })

    def test_invalid_json(self):
        for text in (
            "not-json",
            '```json\n{"readable":false}\n```',
            '{"readable":true,"number":NaN,"background_color":"yellow"}',
            '{"readable":false,"readable":true,"number":14,"background_color":"yellow"}',
        ):
            with self.subTest(text=text):
                result, _ = self.read(text)
                self.assertEqual(result["failure_reason"], "invalid_json")
        result, _ = self.read('{"readable":true,"number":14,"background_color":"yellow","extra":"field"}')
        self.assertEqual(result["failure_reason"], "invalid_schema")

    def test_number_must_be_finite_number_not_bool_or_string(self):
        for number in (True, '14', 'NaN'):
            result, _ = self.read(json.dumps({
                "readable": True, "number": number,
                "background_color": "yellow", "failure_reason": None,
            }))
            self.assertEqual(result["failure_reason"], "invalid_number")
        result, _ = self.read('{"readable":true,"number":null,"background_color":"yellow"}')
        self.assertEqual(result["failure_reason"], "missing_number")
        result, _ = self.read('{"readable":true,"number":1e400,"background_color":"yellow"}')
        self.assertEqual(result["failure_reason"], "invalid_number")

    def test_invalid_color_or_readable_not_success(self):
        for color in (14, [], {}, True, ""):
            with self.subTest(color=color):
                result, _ = self.read(json.dumps({
                    "readable": True, "number": 14,
                    "background_color": color, "failure_reason": None,
                }))
                self.assertFalse(result["readable"])
                self.assertEqual(result["failure_reason"], "invalid_background_color")
        result, _ = self.read('{"readable":"true","number":14,"background_color":"yellow"}')
        self.assertEqual(result["failure_reason"], "invalid_readable")

    def test_contradictory_reading_not_success(self):
        result, _ = self.read('{"readable":true,"number":14,"background_color":"yellow","failure_reason":"unreadable"}')
        self.assertFalse(result["readable"])
        self.assertEqual(result["failure_reason"], "contradictory_reading")

    def test_api_exception_is_safe(self):
        reader = DoubaoNumberColorReader("model-test", client=_RaisingClient())
        result = reader.read(self.image)
        self.assertEqual(result, {
            "readable": False, "number": None,
            "background_color": None, "failure_reason": "TimeoutError",
        })

    def test_missing_image_does_not_call_client(self):
        client = _Client('{"readable":true,"number":1,"background_color":"red"}')
        result = DoubaoNumberColorReader("model-test", client=client).read(Path(self.tmp.name) / "none.png")
        self.assertEqual(result["failure_reason"], "image_not_found")
        self.assertEqual(client.responses.calls, [])


if __name__ == "__main__":
    unittest.main()
