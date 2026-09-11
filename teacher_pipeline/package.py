"""Make the shared merged-base dependency explicit in all pilot adapter artifacts."""
from pathlib import Path
import json
from .common import load_config,parser,write_json,file_digest


def package(c):
    root=Path(c['output']);base=root/'common'/'merged_base';repairs=[]
    for group in ['common/initial_adapter','smoke','runs']:
        for p in (root/group).rglob('adapter_config.json'):
            data=json.loads(p.read_text());old=data.get('base_model_name_or_path')
            if old!=str(base):
                data['base_model_name_or_path']=str(base);write_json(p,data);repairs.append(dict(path=str(p),previous_base=old))
    manifest=dict(shared_base=str(base),base_weights={p.name:file_digest(p) for p in base.glob('*.safetensors')},
                  adapters={str(p.relative_to(root)):file_digest(p) for group in ['common/initial_adapter','runs'] for p in (root/group).rglob('adapter_model.safetensors')},
                  loader='teacher_pipeline.model.load_teacher; local custom Qwen2-VL backbone',
                  pooling='last valid content token, L2 normalization, no added embedding/STEP/LAT token',
                  relocation='Copy common/merged_base together with the chosen adapter+processor; update base path when moving machines.',
                  repaired_base_references=repairs)
    write_json(root/'checkpoint_manifest.json',manifest)
    print('checkpoint package references checked',len(manifest['adapters']))


def main():
    args=parser(__doc__).parse_args();package(load_config(args.config))

if __name__=='__main__':main()
