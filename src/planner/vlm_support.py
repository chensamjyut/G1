from __future__ import annotations
import base64, mimetypes
from pathlib import Path
from typing import Any

def rgb_data_url(rgb: Any) -> str:
    if isinstance(rgb, (str, Path)):
        path = Path(rgb)
        (data, mime) = (path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'image/png')
    elif isinstance(rgb, bytes):
        (data, mime) = (rgb, 'image/png')
    else:
        try:
            from PIL import Image
            import io
            image = Image.fromarray(rgb)
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            (data, mime) = (stream.getvalue(), 'image/png')
        except Exception as exc:
            raise TypeError('RGB must be bytes, a path, or a PIL/numpy image') from exc
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

def response_text(response: Any) -> str:
    output_text = getattr(response, 'output_text', None)
    if output_text:
        return str(output_text)
    output = getattr(response, 'output', None)
    chunks = []
    for item in output or []:
        for content in getattr(item, 'content', []) or []:
            text = getattr(content, 'text', None)
            if text:
                chunks.append(str(text))
    if isinstance(response, dict):
        if response.get('output_text'):
            return str(response['output_text'])
        for item in response.get('output', []):
            for content in item.get('content', []):
                if content.get('text'):
                    chunks.append(str(content['text']))
    return ''.join(chunks) or str(response)
