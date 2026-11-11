#!/usr/bin/env python3
"""Standalone parameter-search table and latency/makespan scatter plot."""
import argparse
import html as html_module
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--confirmation", help="Completed repeated-trial result directory")
    args = parser.parse_args()
    root = Path(args.root)
    data = json.loads((root / "tuning-results.json").read_text())
    html = '''<!doctype html><meta charset="utf-8"><title>15 请求保护参数实验</title>
<style>body{font:15px system-ui;margin:28px;color:#243342}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}svg{max-width:100%;height:auto}.fail{background:#fff0ed}select,label{margin:8px}a{color:#2465a0}</style>
<h1>15 请求保护参数实验</h1>
<p>每个网格点仅运行一次。初筛分数＝short/medium 类 TTFT 均值的几何平均；配对输出必须一致，每条 long E2E ≤ 历史 baseline 的 110%。最终推荐以重复验证报告为准。</p>
<label>显示 <select id="valid"><option value="all">所有点</option><option value="eligible">满足初筛约束</option></select></label>
<label>排序 <select id="sort"><option value="short_medium_geomean_ttft_ms">延迟分数</option><option value="workload_makespan_s">整批请求总时间</option><option value="max_per_request_long_e2e_ratio_vs_baseline">最差 long E2E 比值</option><option value="group_wall_s">启动至清理墙钟</option></select></label>
<p>图：横轴为整批请求总时间（秒），纵轴为延迟分数（秒），越靠左下越小。蓝色满足初筛约束，红色未通过；悬停查看具体参数。</p>
<svg id="plot" viewBox="0 0 1000 420" width="1000" height="420"></svg>
<table><thead><tr><th>参数 c/p/x/w</th><th>准入</th><th>分数(ms)</th><th>最差 long 比值</th><th>整批总时间(s)</th><th>成功req/s</th><th>启动至清理(s)</th><th>抢占chunk</th></tr></thead><tbody id="rows"></tbody></table>
<p>c=chunk blocks，p=抢占次数上限，x=进度阈值，w=累计等待毫秒。准入只是本轮分析约束，不是系统 SLA。group wall 含模型启动和 warmup，不当作推理吞吐分母。</p>
<p><a href="TUNING_REPORT.md">完整报告</a> · <a href="tuning-results.csv">网格 CSV</a> · <a href="tuning-results.json">机器可读结果</a></p>
<script>const data=__DATA__;
function draw(){let rows=data.ranking.filter(r=>document.querySelector('#valid').value==='all'||r.eligible);rows.sort((a,b)=>a[document.querySelector('#sort').value]-b[document.querySelector('#sort').value]);
const body=document.querySelector('#rows');body.replaceChildren();for(const r of rows){let tr=document.createElement('tr');if(!r.eligible)tr.className='fail';let vals=[r.tag,r.eligible?'通过':'未通过',r.short_medium_geomean_ttft_ms.toFixed(1),r.max_per_request_long_e2e_ratio_vs_baseline.toFixed(3),r.workload_makespan_s.toFixed(3),r.successful_requests_per_s.toFixed(3),r.group_wall_s.toFixed(2),r.preemption_count_chunks];for(const v of vals){let td=document.createElement('td');td.textContent=v;tr.append(td)}body.append(tr)}
const svg=document.querySelector('#plot');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';function add(tag,attrs,text){let e=document.createElementNS(ns,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text)e.textContent=text;svg.append(e);return e}
if(!rows.length)return;let xmin=Math.min(...rows.map(r=>r.workload_makespan_s))-.3,xmax=Math.max(...rows.map(r=>r.workload_makespan_s))+.3,ymin=Math.min(...rows.map(r=>r.short_medium_geomean_ttft_ms/1000))-.2,ymax=Math.max(...rows.map(r=>r.short_medium_geomean_ttft_ms/1000))+.2;let x=v=>80+(v-xmin)/(xmax-xmin)*860,y=v=>350-(v-ymin)/(ymax-ymin)*310;
for(let i=0;i<=5;i++){let xv=xmin+(xmax-xmin)*i/5,yv=ymin+(ymax-ymin)*i/5;add('line',{x1:x(xv),x2:x(xv),y1:30,y2:350,stroke:'#e4e8ed'});add('line',{x1:80,x2:940,y1:y(yv),y2:y(yv),stroke:'#e4e8ed'});add('text',{x:x(xv)-10,y:375,'font-size':12},xv.toFixed(1));add('text',{x:40,y:y(yv)+4,'font-size':12},yv.toFixed(1))}add('text',{x:440,y:410,'font-size':14},'整批请求总时间 (s)');add('text',{x:15,y:20,'font-size':14},'延迟分数 (s)');
for(const r of rows){let e=add('circle',{cx:x(r.workload_makespan_s),cy:y(r.short_medium_geomean_ttft_ms/1000),r:6,fill:r.eligible?'#2465a0':'#be4335',opacity:.8});let t=document.createElementNS(ns,'title');t.textContent=r.tag+'; score='+r.short_medium_geomean_ttft_ms+'ms; makespan='+r.workload_makespan_s+'s; long ratio='+r.max_per_request_long_e2e_ratio_vs_baseline;e.append(t)}}document.querySelectorAll('select').forEach(e=>e.addEventListener('change',draw));draw();</script>'''
    if args.confirmation:
        confirmation = Path(args.confirmation)
        decision = json.loads((confirmation / 'final-decision.json').read_text())
        link = html_module.escape(os.path.relpath(confirmation / 'FINAL_TUNING_REPORT.md', root), quote=True)
        recommendation = html_module.escape(str(decision['strict_guardrail_recommendation'] or '无配置通过全部复测约束'))
        banner = f'<p style="padding:14px;background:#fff0ed"><b>重复验证已完成：{recommendation}。</b>初筛最快点未稳定复现；默认也出现输出失败。<a href="{link}">查看最终结论和失败证据</a></p>'
        html = html.replace('<h1>15 请求保护参数实验</h1>', '<h1>15 请求保护参数实验</h1>' + banner)
    (root / "tuning-dashboard.html").write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))


if __name__ == "__main__":
    main()
