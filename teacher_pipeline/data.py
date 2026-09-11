from __future__ import annotations
import re
from pathlib import Path
from PIL import Image
from .common import digest, file_digest

CLEANING_VERSION = 'teacher-native-v1'
LEGACY = re.compile(r'<\|image_\d+\|>|<image>|<video>')


def clean_text(s):
    if not isinstance(s, str):
        raise ValueError('non_string_text')
    return LEGACY.sub('', s).strip()


def content_key(endpoint):
    return {k: endpoint[k] for k in ('text', 'images', 'video_frames')}


def canonical_id(task, endpoint):
    content=content_key(endpoint).copy()
    if task=='Visual7W' and not content['images'] and not content['video_frames']:
        content['text']=re.sub(r'\s+',' ',content['text'].casefold()).strip().rstrip('.!?')
    return digest([task, content])


def endpoint_prompt(endpoint, role):
    # Only this endpoint is accepted; pair labels/answers cannot be forwarded.
    return ('Given only this retrieval endpoint and its task instruction, describe '
            'the observable content and reason about semantic information useful for '
            'retrieval. Preserve attributes, relations and constraints. Do not assume '
            'access to another endpoint, a candidate list or a reference answer. '
            'Do not invent visual evidence. Return concise reasoning grounded in '
            f'this input. Endpoint role: {role}.\nInput: {endpoint["text"]}')


class MediaResolver:
    def __init__(self, roots):
        self.roots = sorted(roots.items(), key=lambda x: -len(x[0]))
        self.cache = {}

    def resolve(self, values):
        if isinstance(values, str):
            values = [values]
        out = []
        for value in values or []:
            if not value or value.replace('\\', '/').strip('./') == 'images/blank.jpg':
                continue
            for prefix, root in self.roots:
                if value.startswith(prefix):
                    p = (Path(root) / value[len(prefix):]).resolve()
                    if not p.is_relative_to(Path(root).resolve()):
                        raise ValueError('unsafe_media_path')
                    break
            else:
                raise ValueError('unmapped_media')
            if not p.is_file():
                raise ValueError('missing_media')
            if str(p) not in self.cache:
                try:
                    with Image.open(p) as image:
                        image.load()
                        # Hash decoded pixels so identical media at different paths group together.
                        im = image.convert('RGB')
                        self.cache[str(p)] = digest([im.size, __import__('hashlib').sha256(im.tobytes()).hexdigest()])
                except Exception as e:
                    raise ValueError('unreadable_media') from e
            out.append(str(p))
        return out


def normalize(raw, task, index, resolver):
    if raw.get('error'):
        raise ValueError('source_error')
    endpoints = []
    for side, cot_key in [('query', 'query_cot'), ('candidate', 'pos_cot')]:
        is_q = side == 'query'
        source = raw.get('qry' if is_q else 'pos')
        if isinstance(source, dict):
            kind = raw.get('dataset_name', '')
            if kind not in {'llavahound_video_retrieval', 'llavahound_caption_retrieval', 'llavahound_qa'}:
                raise ValueError('unknown_video_task')
            role = ('gpt' if is_q else 'human') if kind == 'llavahound_video_retrieval' else ('human' if is_q else 'gpt')
            text = next((v['value'] for v in source.get('conversations', []) if v.get('from') == role), '')
            images = resolver.resolve(source.get('image', []))
            frames = resolver.resolve(source.get('video', []))
        else:
            text = raw.get('qry' if is_q else 'pos_text', '')
            images = resolver.resolve(raw.get('qry_image_path' if is_q else 'pos_image_path', []))
            frames = []
        text = clean_text(text)
        cot = raw.get(cot_key, '')
        if not isinstance(cot, str) or not re.sub(r'<[^>]+>', '', cot).strip():
            raise ValueError('empty_cot')
        if not (text or images or frames):
            raise ValueError('empty_endpoint')
        ep = dict(text=text, images=images, video_frames=frames, cot=cot.strip(), cot_provenance='unknown')
        ep['id'] = canonical_id(task, ep)
        endpoints.append(ep)
    q, c = endpoints
    if task in {'MSCOCO_i2t','Visual7W'} and not q['images']:
        raise ValueError('missing_expected_query_media')
    media = sorted({resolver.cache[p] for ep in endpoints for p in ep['images'] + ep['video_frames']})
    # Query identity links multiple answers. Shared answer words never join groups.
    group_keys = ['media:' + k for k in media] + ['query:' + digest(content_key(q))]
    flags = []
    for side, ep in zip(('query','candidate'), endpoints):
        if not all(t in ep['cot'] for t in ('<thinking>', '<rethink>', '<answer>')):
            flags.append(side + '_nonstandard_tags')
        if re.search(r'\b(?:the query|the question|correct output|positive (?:text|answer)|matches? the (?:query|question))\b', ep['cot'], re.I):
            flags.append(side + '_possible_pair_conditioning')
    return dict(sample_id=f'{task}.json:{index}', dataset=task, task=task,
                source_group_id=None, group_keys=group_keys, query=q, candidate=c,
                positive_candidate_ids=[c['id']], quality_flags=flags,
                original_row_id=index, source_file=f'{task}.json', cleaning_version=CLEANING_VERSION)


class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, x):
        p = self.parent.setdefault(x, x)
        if p != x: self.parent[x] = self.find(p)
        return self.parent[x]
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        self.parent[max(a,b)] = min(a,b)
