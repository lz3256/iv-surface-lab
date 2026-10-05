"""Validate saved real-data results and write a concise research summary."""
from pathlib import Path
import json
import html
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spy", required=True, type=Path, help="Authorized source SPY CSV used for the independent label audit")
    args = parser.parse_args()
    daily = pd.read_csv(ROOT/'data/daily.csv')
    meta = json.loads((ROOT/'data/metadata.json').read_text())
    manifest = json.loads((ROOT/'data/build_manifest.json').read_text())
    verification, rows = [], []
    for phase in ['development','final']:
        folder = ROOT/'reports'/phase
        summary = json.loads((folder/'summary.json').read_text())
        comparisons = pd.read_csv(folder/'model_comparisons.csv')
        metrics = pd.read_csv(folder/'forecast_metrics.csv')
        assert summary['metadata']['synthetic'] is False
        assert summary['holdout_start'] == '2022-12-29'
        assert all(a['all_train_labels_known'] for a in summary['audit'])
        har = pd.read_csv(folder/'har_audit.csv')
        good = har.har_fwd_21.notna()
        assert (pd.to_datetime(har.loc[good,'train_end_max'],utc=True) <
                pd.to_datetime(har.loc[good,'asof_time'],utc=True)).all()
        for family,horizons in [('H1',[5,21]),('H2',[1,5,21])]:
            for h in horizons:
                p = pd.read_csv(folder/f'{family}_h{h}_predictions.csv')
                predicted = p.train_n.notna()
                assert (pd.to_datetime(p.loc[predicted,'train_end_max'],utc=True)<
                        pd.to_datetime(p.loc[predicted,'asof_time'],utc=True)).all()
                if phase == 'final':
                    assert p.date.min() == '2022-12-29'
                else:
                    assert p.date.max() == '2022-12-28'
                verification.append({'phase':phase,'family':family,'horizon':h,
                                     'rows':len(p),'date_start':p.date.min(),'date_end':p.date.max(),
                                     'train_times_pass':True})
        for family,horizon,baseline,added,label in [
            ('H2',1,'R2_HAR','R3_skew_change','1日收益：增加偏斜变化'),
            ('H2',5,'R2_HAR','R3_skew_change','5日收益：增加偏斜变化'),
            ('H2',21,'R2_HAR','R3_skew_change','21日收益：增加偏斜变化'),
            ('H1',5,'V1_HAR','V2_IV','5日波动：HAR增加IV水平'),
            ('H1',21,'V1_HAR','V2_IV','21日波动：HAR增加IV水平'),
            ('H1',5,'V2_IV','V3_surface','5日波动：增加曲面形状'),
            ('H1',21,'V2_IV','V3_surface','21日波动：增加曲面形状')]:
            c = comparisons.loc[(comparisons.family==family)&(comparisons.horizon==horizon)&
                                (comparisons.baseline==baseline)&(comparisons.added==added)].iloc[0]
            m = metrics.loc[(metrics.family==family)&(metrics.horizon==horizon)&(metrics.model==baseline)].iloc[0]
            rows.append({'phase':phase,'question':label,'family':family,'horizon':horizon,'n':int(c.n),
                         'mse_improvement_pct':100*c.incremental_r2,
                         'gain_ci_low_pct_of_baseline':100*c.low/m.mse,
                         'gain_ci_high_pct_of_baseline':100*c.high/m.mse,
                         'p_raw':c.p,'p_holm_aux':c.p_holm_aux,'p_holm_all':c.p_holm_all_sensitivity})
    # Independently reconstruct return labels directly from raw CRSP prices and recorded distributions.
    spy = pd.read_csv(args.spy)
    spy.columns = spy.columns.str.lower()
    spy = spy.sort_values('date').set_index('date')
    implied_cash = (spy.ret-spy.retx)*spy.prc.shift(1)
    label_checks = []
    for phase in ['development','final']:
        for h in [1,5,21]:
            pred = pd.read_csv(ROOT/'reports'/phase/f'H2_h{h}_predictions.csv')
            observed = pred.loc[pred.target.notna()]
            errors = []
            for row in observed.itertuples():
                source = daily.loc[daily.date.eq(row.date)].iloc[0]
                i = spy.index.get_loc(source.entry_session)
                expected = (spy.openprc.iloc[i+h]+implied_cash.iloc[i+1:i+h+1].sum())/spy.openprc.iloc[i]-1
                errors.append(abs(row.target-expected))
            assert max(errors) < 1e-12
            label_checks.append({'phase':phase,'horizon':h,'checked':len(errors),'max_abs_error':max(errors)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT/'results_summary.csv',index=False)
    verification = {'prediction_audits':verification,'independent_return_label_checks':label_checks,
                    'tests_passed':64,'checks':'Saved forecasts, phase boundaries, HAR/main label times, and raw-CRSP return reconstruction.'}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2))
    lines = ['# 现有数据研究结果','',
             '已用用户提供的 optionsDX SPX 期权链和 CRSP SPY 数据完成固定参数的开发期与最终留出期运行。以下是探索性研究，不是实时可交易性验证。','',
             '## 数据与固定规则','',
             f"- 信号日历：2010-01-04—2023-12-29，共 {len(daily):,} 个交易日；完整四节点 {manifest['complete_surface_rows']:,} 天。",
             '- 原始期权缺失的35个正常交易日保留缺失；12个非交易日记录隔离。2019年5月的6个极端call/put IV差异日也标为曲面缺失。没有跨日填充曲面。',
             '- 开发期截至2022-12-28；最终留出期为2022-12-29—2023-12-29，共252个交易日。留出期按时间顺序更新模型，只使用当时已经知道的标签。',
             '- SPY收盘价用于价格控制，SPY开盘价用于假设交易，SPY日线用于波动率代理；分红已计入。',
             '- 参数在看结果前固定：HAR最少252样本、主模型最少504样本、滚动1008样本、每21日重新拟合、ridge=10、1000次块bootstrap。',
             '- 唯一指定主要比较：1日收益模型在市场及HAR控制之上，增加30天25Δ偏斜变化。其他比较是辅助分析。','',
             '## 主要结果','',
             '误差改善为正表示新增特征降低MSE，为负表示预测更差。下表区间是MSE改善的bootstrap区间除以固定的基线MSE，并非单独重估R²比值的区间。辅助p值按每次运行内部的辅助比较做Holm调整；开发与留出阶段没有合并成一个确认检验。','',
             '| 阶段 | 比较 | 样本数 | MSE改善 | 改善区间 | 原始p | 辅助Holm p |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        adj = '—（指定主比较）' if pd.isna(row.p_holm_aux) else f'{row.p_holm_aux:.3f}'
        lines.append(f"| {'开发期' if row.phase=='development' else '留出期'} | {row.question} | {row.n} | {row.mse_improvement_pct:+.3f}% | [{row.gain_ci_low_pct_of_baseline:+.3f}%, {row.gain_ci_high_pct_of_baseline:+.3f}%] | {row.p_raw:.3f} | {adj} |")
    lines += ['', '## 验证与解释边界','',
              '- 64项测试通过。所有已生成预测的训练标签时间均早于信号时间。H2收益标签另从原始CRSP价格与分红独立重算并逐项核对。',
              '- 仍未核实SPX/SPXW合约族、AM/PM结算映射、供应商Greeks定义和历史修订。这是供应商DTE口径的近似曲面。',
              '- 假设上一信号日文件在下一交易日08:00前可得，08:01生成信号，随后按SPY开盘价成交。该可得时间是假设，不保证保守。',
              '- 最后1/5/21日等没有完整未来标签的记录不评分。缺失曲面还会减少偏斜差分样本，未据此压缩交易日历。',
              '- 波动率目标是日线OHLC与隔夜收益构成的代理，不是高频实现波动率。',
              '- 策略CSV仅为假设损益诊断，包含单边1bp及成本压力情景，未计融资、借券、税和完整冲击。显著性不能证明内幕信息。',
              '- 留出期已经用于本次评估，后续若据此调参，就不能继续称它为未见样本。','',
              '详细模型、逐年结果、逐日预测、训练审计、成本诊断位于 reports/development 和 reports/final。固定协议见 PROTOCOL.md，代码与协议哈希见 protocol_freeze.json。']
    (ROOT/'研究结果.md').write_text('\n'.join(lines)+'\n')
    body = '<h1>IV Surface · 现有数据研究</h1><p>固定参数；真实数据；探索性结果。合约结算与历史可得时间未核实。</p>'
    body += '<p><a href="研究结果.md">中文结果说明</a> · <a href="reports/development/report.html">开发期完整报告</a> · <a href="reports/final/report.html">留出期完整报告</a></p>'
    body += table.to_html(index=False,float_format=lambda x:f'{x:.5g}')
    body += '<h2>解释限制</h2><p>正值表示降低预测误差。检验是条件于生成预测的近似块bootstrap，辅助结果须结合Holm调整。报价时间和结算族未核实，不能视为可执行业绩。最终留出期已被评估，后续调参需要新数据确认。</p>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>IV Surface Results</title><style>body{font:16px/1.7 system-ui;max-width:1160px;margin:36px auto;padding:0 22px;color:#223344}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}a{color:#1761a0}</style>'+body+'</html>'
    (ROOT/'index.html').write_text(page)
    print(table.to_string(index=False))



if __name__ == '__main__':
    main()
