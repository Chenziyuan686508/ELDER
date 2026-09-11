"""Finish this bounded pilot sequentially after the active A run; stop on first failure."""
import json
import subprocess
import sys
import time
from pathlib import Path
from .common import parser,load_config,write_json


def main():
    args=parser(__doc__).parse_args();c=load_config(args.config);root=Path(c['output'])
    deadline=time.monotonic()+3600
    while not (root/'runs'/'no_cot'/'result.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('A completion wait timed out; inspect train_a.log')
        time.sleep(5)
    def run(module,extra=(),python=sys.executable,log=None):
        command=[python,'-m','teacher_pipeline.'+module,'--config',args.config,*extra]
        print('RUN',command,flush=True)
        with (root/(log or module+'.log')).open('a') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        # The pause/resume is a real optimizer continuation, with an unchanged plan/fingerprint.
        if not (root/'runs'/'both_cot'/'result.json').exists():
            if not (root/'runs'/'both_cot'/'latest.json').exists():
                run('train',['--mode','both_cot','--stop-after-steps','10'],log='train_b.log')
            run('train',['--mode','both_cot','--resume'],log='train_b.log')
        run('package')
        run('evaluate',['--suite','base,a,b_off'],log='evaluate_no_cot.log')
        run('generate_eval_cot',python='/root/miniconda3/envs/elder-cot/bin/python',log='generate_eval_cot.log')
        run('evaluate',['--suite','b'],log='evaluate_b.log')
        run('process_export',['--enable'],log='process_export.log')
        write_json(root/'execution_status.json',dict(status='completed'))
    except subprocess.CalledProcessError as e:
        write_json(root/'execution_status.json',dict(status='failed',command=e.cmd,exit_code=e.returncode))
        raise
    finally:run('report')

if __name__=='__main__':main()
