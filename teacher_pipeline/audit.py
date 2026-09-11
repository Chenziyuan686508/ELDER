"""Record actual environment and local model fingerprints; full data audit is prepare."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from .common import parser,load_config,write_json,file_digest


def main():
    args=parser(__doc__).parse_args();c=load_config(args.config);out=Path(c['output'])
    packages={}
    for name in ['torch','transformers','peft','Pillow','huggingface-hub','pytest','PyYAML','flash-attn','numpy','accelerate','safetensors','tokenizers','packaging']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    files={}
    for key in ['backbone','name_or_path']:
        p=Path(c['model'][key]);files[key]={str(f.name):file_digest(f) for f in sorted(p.glob('*')) if f.is_file() and f.suffix in {'.json','.safetensors','.bin'}}
    report=dict(python=platform.python_version(),packages=packages,gpu=gpu.stdout,gpu_error=gpu.stderr,commit=commit,
                dirty_worktree=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
                disk_free_bytes=shutil.disk_usage(out).free,model_files=files,model=c['model'])
    write_json(out/'environment.json',report)
    (out/'environment_report.md').write_text('# Teacher pilot environment\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n\nThe base files are identified by SHA256; local base HF revision has not been inferred.\n')
    Path('requirements/teacher-pilot.lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in packages.items() if v)+'\n')
    print(report['gpu']);print(out/'environment_report.md')

if __name__=='__main__': main()
