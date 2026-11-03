"""One real current-frame request in the isolated Ark SDK environment."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner.number_color_reader import DoubaoNumberColorReader

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    reader = DoubaoNumberColorReader(a.model)
    result = (reader.read(a.image) if os.getenv('ARK_API_KEY') else
              reader._failure('missing_ARK_API_KEY'))
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
