from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    if c['training']['negative_pool_backend']!='gradcache' or c['training']['gradient_accumulation_steps']!=1:
        raise ValueError('This pilot implements local exact GradCache with accumulation=1 only')
    if c['training']['max_epochs']!=1 or c['model']['pooling']!='last_valid_token':
        raise ValueError('This pilot requires one epoch and native last-valid-token pooling')
    if not c['model']['freeze_vision'] or not c['model']['freeze_visual_connector']:
        raise ValueError('This pilot only supports frozen vision/connector')
    c['config_path'] = str(Path(path).resolve())
    c['output'] = str(Path(c['experiment']['output_dir']))
    Path(c['output']).mkdir(parents=True, exist_ok=True)
    return c


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default='configs/teacher_pilot.yaml')
    return p


def iter_json_array(path, chunk_size=65536):
    """Strict, bounded-memory streaming JSON array reader, no optional dependency."""
    decoder = json.JSONDecoder()
    with open(path, encoding='utf-8-sig') as f:
        buf, eof = '', False
        def fill():
            nonlocal buf, eof
            block = f.read(chunk_size)
            buf += block
            eof = not block
        def space():
            nonlocal buf
            buf = buf.lstrip()
            while not buf and not eof:
                fill()
                buf = buf.lstrip()
        fill(); space()
        if not buf.startswith('['):
            raise ValueError(f'{path}: expected JSON array')
        buf = buf[1:]
        first = True
        while True:
            space()
            if buf.startswith(']'):
                buf = buf[1:]
                while not eof:
                    fill()
                if buf.strip():
                    raise ValueError('Trailing content after array')
                return
            if not first:
                if not buf.startswith(','):
                    raise ValueError('Missing array comma or closing bracket')
                buf = buf[1:]; space()
            while True:
                try:
                    value, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    if len(buf)>8*1024*1024:
                        raise ValueError('Malformed or oversized JSON record (>8 MiB characters)')
                    fill()
            if not isinstance(value, dict):
                raise ValueError('Expected object record')
            yield value
            buf = buf[end:]
            first = False
