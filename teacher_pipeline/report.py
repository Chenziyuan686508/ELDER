"""Write an evidence-only progress report from completed artifacts."""
from pathlib import Path
import json
from .common import parser,load_config


def main():
    args=parser(__doc__).parse_args();c=load_config(args.config);root=Path(c['output'])
    def read(path):
        p=root/path
        return json.loads(p.read_text()) if p.exists() else None
    audit=read('data_audit.json');init=read('common/initialization.json');smoke=read('smoke/report.json')
    lines=['# 双端 CoT 检索教师 pilot','',
        '执行规格：教师模型训练执行说明.md（2026-09-07）。本实验独立于旧 ELDER Stage 1/2/3。',
        '固定官方 VLM2Vec-V2.0 adapter revision：`'+c['model']['revision']+'`。',
        '将 released DoRA 合并后，新建语言侧 rank-32 LoRA；A/B 读取同一公共 adapter 初始化与批次计划。',
        'Base 读取原始 released adapter；A/B 的合并数值误差单独记录。没有新增 STEP/LAT token，没有 LM 或过程蒸馏损失。','']
    if init: lines += [f"可训练参数：{init['trainable_parameters']:,}；视觉冻结：{init['visual_frozen']}；文本合并最大绝对误差：{init['merge_max_abs']:.8f}。",'']
    if audit:
        lines+=['## 数据','', '```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','']
    lines+=['## 工程验证','', 'CPU 数值测试见实际 pytest 日志 `tests.log`；含多正例 loss、无正例屏蔽、GradCache loss/梯度/RNG 一致性和完整候选库指标。']
    if smoke:lines+=['','```json',json.dumps(smoke,indent=2),'```']
    else:lines+=['冒烟尚未完成。']
    budget=read('input_budget_audit.json')
    if budget:lines+=['',f"共同输入预算审计：{budget['retained_train_pairs']}/{budget['scanned_train_pairs']} 训练对可用，统计 {budget['statistics']}。"]
    lines+=['','## 训练与结果','']
    for mode in ['no_cot','both_cot']:
        result=read('runs/'+mode+'/result.json')
        if result:
            metrics=[json.loads(l) for l in (root/'runs'/mode/'metrics.jsonl').open()]
            lines += [f"- {mode}：{result['optimizer_steps']} 步；重载误差 {result['reload_max_abs']}；模型计算累计 {sum(x['seconds'] for x in metrics):.1f} 秒（不含 processor/保存）；输入 token {sum(x['input_tokens'] for x in metrics):,}；峰值分配显存 {max(x['peak_memory_bytes'] for x in metrics)/2**30:.2f} GiB。"]
        else:lines+=[f'- {mode}：尚无已完成训练/重载验收结果。']
    groups={g:read('evaluation/'+g+'.json') for g in ['base','a','b','b_off']}
    lines+=['','| 组 | 任务 | query / candidate | Hit@1 | Recall@5 |','|---|---|---|---|---|']
    for g,result in groups.items():
        if result:
            for task,v in result.items():lines.append(f"| {g} | {task} | {v['queries']} / {v['candidates']} | {100*v['metrics']['hit1']:.2f} | {100*v['metrics']['recall5']:.2f} |")
            values=list(result.values())
            lines.append(f"| {g} | 两任务等权宏平均（Image） | — | {100*sum(v['metrics']['hit1'] for v in values)/len(values):.2f} | {100*sum(v['metrics']['recall5'] for v in values)/len(values):.2f} |")
        else:lines.append(f'| {g} | 未完成 | — | — | — |')
    delta=read('evaluation/b_minus_a.json')
    if delta:lines+=['','B−A 配对媒体组 bootstrap（比例单位）：','```json',json.dumps(delta,indent=2),'```']
    if all(groups.values()):
        tasks=c['data']['selected_tasks']
        change=100*sum(groups['b'][t]['metrics']['hit1']-groups['a'][t]['metrics']['hit1'] for t in tasks)/len(tasks)
        lines+=['',f'正常 CoT 条件下，B−A 两任务等权 Hit@1 差值为 {change:+.2f} 个百分点。']
        if delta and all(v['ci95'][0]>0 for v in delta.values()):
            lines+=['两任务的配对区间均高于零；仅作为本 pilot 的继续验证依据，尚不能推广到完整 benchmark 或学生蒸馏。']
        else:
            lines+=['尚不能声称所有任务都获得可信收益；结合逐任务区间、B-off 和错误样例判断。下一项优先排查是训练特权 CoT 与独立生成 CoT 的分布差异，不做无界调参。']
    if not all(groups.values()):lines+=['','**正常 held-out 对照尚未完成，不能判断 CoT 收益，不能用训练 CoT 诊断值替代。**']
    lines+=['','## 限制与状态','',
        '- 这是两任务 grouped-holdout pilot，候选库为完整的过滤后 held-out 候选清单，不是官方 MMEB-V2 全量成绩。',
        '- A/B 为相同数据和更新预算，不是相同 FLOPs；负例池按真实 query 数及去重候选数报告。',
        '- 公共训练数据的 CoT provenance 保持 unknown。候选 CoT 采用保守词法风险筛查，再按 CoT SHA256 确定复用版本；这不能证明语义上绝无配对依赖。',
        '- 随机抽查每任务 20 条，记录见 review/review_notes.json；这是助手文本检查，不是人工视觉忠实度评分。已记录并剔除两条明显矛盾。',
        '- 视频、视觉文档、组合图像检索未纳入本 pilot；不宣称这些模态已通过真实冒烟。',
        '- 一次前向 latent student 与过程蒸馏不在本次实现范围。可选状态导出的实际完成情况见 process_export/ 与执行日志。',
        '- 单种子 bootstrap 不能替代至少三个种子的复现。','',
        '## 命令','',
        '在 /root/code/ELDER 使用 /root/miniconda3/envs/elder/bin/python；生成器命令改用 elder-cot 环境。',
        '```bash',
        'python -m teacher_pipeline.audit --config configs/teacher_pilot.yaml',
        'python -m teacher_pipeline.prepare --config configs/teacher_pilot.yaml',
        'python -m teacher_pipeline.preflight --config configs/teacher_pilot.yaml',
        'python -m teacher_pipeline.smoke --config configs/teacher_pilot.yaml',
        'python -m teacher_pipeline.train --config configs/teacher_pilot.yaml --mode no_cot',
        'python -m teacher_pipeline.train --config configs/teacher_pilot.yaml --mode both_cot',
        'python -m teacher_pipeline.generate_eval_cot --config configs/teacher_pilot.yaml',
        'python -m teacher_pipeline.evaluate --config configs/teacher_pilot.yaml --suite base,a,b,b_off',
        'python -m teacher_pipeline.report --config configs/teacher_pilot.yaml','```','',
        '训练恢复加 --resume；各组输出位于 runs/no_cot 与 runs/both_cot。已执行记录以对应日志和 JSON 为准，以上是命令接口。']
    (root/'teacher_report.md').write_text('\n'.join(lines)+'\n');print(root/'teacher_report.md')

if __name__=='__main__':main()
