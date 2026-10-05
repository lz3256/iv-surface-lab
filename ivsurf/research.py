"""H1 volatility, H2 returns and explicitly exploratory event analyses."""
from __future__ import annotations
import hashlib
import html
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from . import __version__
from . import core
from .data import read, save_json


def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codehash():
    h=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def inference_families(comparison, kind, primary_eligible=True):
    """Fixed primary definition. Registration is a research protocol, not a p-value choice."""
    out = comparison.copy()
    primary = ((out.family=='H2') & (out.horizon==1) &
               (out.baseline=='R2_HAR') & (out.added=='R3_skew_change') & (kind=='curve') & primary_eligible)
    if int(primary.sum()) > 1:
        raise ValueError('Duplicate primary comparison.')
    out['test_role'] = np.where(primary,'primary','auxiliary')
    out['p_primary'] = out.p.where(primary)
    out['p_holm_aux'] = np.nan
    out.loc[~primary,'p_holm_aux'] = core.holm(out.loc[~primary,'p'])
    out['p_holm_all_sensitivity'] = core.holm(out.p)
    return out


def specs(kind):
    market=['ret_1','ret_5','rv_22','iv_var_30','term_slope']
    controls=market+['har_fwd_21']
    change=['skew_change'] if kind=='curve' else ['skew_index_change']
    shape=['term_slope','skew','curvature'] if kind=='curve' else ['term_slope','skew_index','vvix_ratio']
    h2={'R0_mean':[], 'R1_market':market, 'R2_HAR':controls,
        'R3_skew_change':controls+change, 'R4_VRP_only':['vrp_proxy']}
    har=['log_rv_1','log_rv_5','log_rv_22']
    h1={'V0_mean':[], 'V1_HAR':har, 'V2_IV':har+['log_iv_var_30'],
        'V3_surface':har+['log_iv_var_30']+shape}
    return h1,h2


def metrics(pred, names, family, horizon, reps, block):
    rows=[]; comparisons=[]; yearly=[]
    common=pred.dropna(subset=['target']+names)
    if len(common)<30:
        raise ValueError(f'{family}, h={horizon}: fewer than 30 common OOS targets. Need more complete data.')
    y=common.target
    base=names[0]
    base_mse=((y-common[base])**2).mean()
    for name in names:
        error=(y-common[name])**2
        rows.append({'family':family,'horizon':horizon,'model':name,'n':len(common),
                     'mse':float(error.mean()),'r2_vs_mean':float(1-error.mean()/base_mse),
                     'direction_accuracy':float(((common[name]>0)==(y>0)).mean()) if family=='H2' else np.nan})
    pairs=[('V2_IV','V3_surface'),('V1_HAR','V2_IV')] if family=='H1' else [
        ('R2_HAR','R3_skew_change'),('R1_market','R2_HAR'),('R0_mean','R4_VRP_only')]
    for baseline,added in pairs:
        test=core.block_test(pred[baseline],pred[added],pred.target,reps,max(block,horizon))
        a=((y-common[baseline])**2).mean(); b=((y-common[added])**2).mean()
        comparisons.append({'family':family,'horizon':horizon,'baseline':baseline,'added':added,
                            'incremental_r2':float(1-b/a),**test,
                            'cw_t_descriptive':core.clark_west(pred.target,pred[added],pred[baseline],max(5,horizon))})
    primary=pairs[0]
    for year,g in common.groupby(common.date.str[:4]):
        a=((g.target-g[primary[0]])**2).mean(); b=((g.target-g[primary[1]])**2).mean()
        yearly.append({'family':family,'horizon':horizon,'year':year,'n':len(g),
                       'incremental_r2':float(1-b/a),'mse_gain':float(a-b)})
    return rows,comparisons,yearly


def forecast_groups(pred,name):
    history=[]; buckets=[]
    for x in pred[name]:
        if np.isfinite(x) and len(history)>=60:
            cuts=np.quantile(history[-252:],[.2,.4,.6,.8])
            buckets.append(int(np.searchsorted(cuts,x,side='right')+1))
        else:
            buckets.append(np.nan)
        if np.isfinite(x):
            history.append(x)
    d=pred.copy(); d['group']=buckets
    return d.dropna(subset=['group','target']).groupby('group').agg(n=('target','size'),
                 mean_return=('target','mean'),mean_prediction=(name,'mean')).reset_index()


def event_analysis(panel,pred,events,kind):
    e=events.copy()
    core.require(e,['event_id','event_type','event_time'])
    if e.event_id.duplicated().any():
        raise ValueError('event_id must be unique.')
    e['event_time']=core.times(e.event_time,'event_time')
    d=panel.loc[pred.index].copy()
    d['target']=pred.target
    d['target_end']=pred['target_exit_time'] if 'target_exit_time' in pred else pred.target_end
    counts=[]; regressions=[]
    change='skew_change' if kind=='curve' else 'skew_index_change'
    for etype,group in e.groupby('event_type'):
        during=np.zeros(len(d),dtype=bool); before=np.zeros(len(d),dtype=bool)
        in_ids=set()
        for event in group.itertuples():
            flag=(d.entry_time < event.event_time)&(event.event_time<=d.target_end)
            during |= flag.to_numpy()
            before |= ((d.asof_time < event.event_time)&(event.event_time<=d.entry_time)).to_numpy()
            if flag.any():
                in_ids.add(event.event_id)
        for label,mask in [('in_holding_window',during),('pre_entry_only',before&~during),('neither',~before&~during)]:
            yy=d.loc[mask,'target'].dropna()
            counts.append({'event_type':etype,'group':label,'n_sessions':len(yy),
                           'distinct_in_window_events':len(in_ids),'mean_return':yy.mean(),'mean_abs_return':yy.abs().mean()})
        # Descriptive interaction with the same controls; no causal/private-information claim.
        names=['ret_1','ret_5','rv_22','iv_var_30','term_slope','har_fwd_21',change]
        x=d[names].copy()
        x['event']=during.astype(float); x['event_x_change']=x[change]*during
        if during.sum()>=10 and (~during).sum()>=10:
            reg=core.hac_ols(d.target,x.to_numpy(),list(x.columns),lags=5,positions=d.session_id)
            reg['event_type']=etype
            regressions.append(reg)
    return pd.DataFrame(counts),pd.concat(regressions,ignore_index=True) if regressions else pd.DataFrame()


def make_html(directory,summary,tables):
    label='模拟数据：仅验证代码。' if summary['metadata']['synthetic'] else '用户数据：来源与实际可得时间仍需核实。'
    body=f'<h1>IV Surface Lab · Unified v{__version__}</h1><aside>{label}</aside>'
    body+='<p>唯一指定主比较：真实曲面H2、1日、R3_skew_change 相对 R2_HAR。H1、其他期限与模型比较均为辅助；Cboe代理模式没有原曲面主检验。是否事前注册需研究者自行证明。</p>'
    body+='<p>HAR预测方差已进入R2/R3。VRP代理单独检验，因为它等于隐含方差减HAR；把三者同时加入会精确共线。标准化曲面期限为30/91日，Cboe代理采用VIX与VIX3M，不将3M改名为91天。HAR为21交易日，VRP是近似期限匹配的代理。</p>'
    body+='<p>gain为基线MSE减增强模型MSE，正数代表改善。low/high是逐项bootstrap近似95%区间，并非多重校正区间。p_primary为唯一指定主比较的原始双侧p；p_holm_aux只校正辅助比较；p_holm_all_sensitivity保留全部比较校正供敏感性核对。此分组不控制主加辅助合计的整体错误率，也不覆盖跨运行调参。CW仅为描述性统计。</p>'
    body+='<p>新版WRDS输入包含现金分配：标签累加入场日之后至退出日的分红，入场日分红不计；隔夜平方收益同步校正。RET/RETX推导的分红到当日收盘后才视为已知，标签可得时间与实际退出时间分开。旧输入未带分红时仍是价格收益。</p>'
    body+='<p>多日策略采用1/h交错子组合、逐日盯市及仓位缩放；有分红输入时日收益包含应得分红（空头承担支付）。多日标签是持有现金、不再投资的总收益，诊断策略按日复合收益，两者投资规则不同。未计支付延迟、税、换月、融资和融券。除息剔除仅用于回顾性标签敏感性，不是用未来分红安排的避交易策略。</p>'
    body+='<h2>运行记录</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2,default=str))+'</pre>'
    for name,table in tables.items():
        body+='<h2>'+html.escape(name)+'</h2>'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')
    body+='<p>逐日预测、训练标签截止时间、HAR审计和逐日策略损益见同目录CSV。事件分析为探索性日频分组，不能证明内幕消息；公布前但入场前的事件单独计数。</p>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>IV Surface Lab</title><style>body{font:15px/1.65 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#173047}aside{background:#fff0cc;border-left:4px solid #bd8210;padding:16px}table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:8px 12px;border-bottom:1px solid #dce4ed;text-align:right}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f1f5f8;padding:16px;font-size:12px}h2{margin-top:32px}</style>'''+body+'</html>'
    (directory/'report.html').write_text(page,encoding='utf-8')


def run(args):
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty. Use a new run directory to preserve experiment history.')
    if args.min_train<30 or args.har_min<30 or args.window<max(args.min_train,args.har_min) or args.refit<1 or args.alpha<=0:
        raise ValueError('Invalid training settings: min_train/har_min>=30, window>=both, refit>=1, alpha>0.')
    if args.holdout<30 or args.bootstrap<100 or args.block<1 or args.cost_bps<0:
        raise ValueError('Require holdout>=30, bootstrap>=100, block>=1, nonnegative cost.')
    for horizons in [args.return_horizons,args.vol_horizons]:
        if not horizons or min(horizons)<1 or len(set(horizons))!=len(horizons):
            raise ValueError('Each horizon list must contain distinct positive integers.')
    meta=json.loads(Path(args.metadata).read_text())
    raw=core.validate(read(args.input),read(args.calendar),meta)
    exclude_ex=getattr(args,'exclude_ex_dividend',False)
    sensitivity=getattr(args,'study_role','primary')=='sensitivity' or exclude_ex or args.window>=1_000_000
    if exclude_ex and 'entry_is_ex_dividend' not in raw:
        raise ValueError('--exclude-ex-dividend requires validated dividend fields; rebuild WRDS inputs.')
    boundary=len(raw)-args.holdout
    if boundary<=args.min_train+args.har_min+45:
        raise ValueError('Not enough development sessions for nested HAR + return training. Add history or reduce windows for a smoke test.')
    cutoff=raw.asof_time.iloc[boundary]
    subset=raw.iloc[:boundary].copy() if args.phase=='development' else raw.copy()
    panel=core.features(subset,meta['surface_kind'])
    panel,har=core.add_har(panel,args.har_min,args.window,args.refit,args.alpha)
    development=panel.loc[panel.asof_time<cutoff]
    coverage={}
    if meta['surface_kind']=='curve':
        coverage={'complete_surface_fraction':float(development[core.SURFACE].notna().all(axis=1).mean()),
                  'skew_change_fraction_excluding_first':float(development.skew_change.iloc[1:].notna().mean()),
                  'sessions':len(development),'imputation':'none'}
    h1,h2=specs(meta['surface_kind'])
    out.mkdir(parents=True,exist_ok=True)
    panel.to_csv(out/'features.csv',index=False)
    har.to_csv(out/'har_audit.csv',index=False)
    all_metrics=[]; all_comparisons=[]; all_yearly=[]; strategies=[]; audit=[]; diagnostics=[]
    event_tables={}; group_tables={}
    for family,kind,horizons,models in [('H1','vol',args.vol_horizons,h1),('H2','return',args.return_horizons,h2)]:
        for horizon in horizons:
            y,end=core.target(panel,kind,horizon)
            excluded=core.ex_dividend_window(panel,horizon) if family=='H2' and exclude_ex else pd.Series(False,index=panel.index)
            y=y.mask(excluded)
            pred=core.walk(panel,y,end,models,args.min_train,args.window,args.refit,args.alpha)
            if family=='H2':
                pred['target_exit_time']=panel.entry_time.shift(-horizon)
                pred['ex_dividend_label_excluded']=excluded
            if args.phase=='final':
                ev=pred.iloc[boundary:].copy()
            else:
                first=pred[list(models)[0]].first_valid_index()
                if first is None:
                    raise ValueError(f'{family} h={horizon}: no forecast; inspect feature coverage and training length.')
                ev=pred.loc[first:].copy()
            met,comp,annual=metrics(ev,list(models),family,horizon,args.bootstrap,args.block)
            all_metrics+=met; all_comparisons+=comp; all_yearly+=annual
            ev.to_csv(out/f'{family}_h{horizon}_predictions.csv',index=False)
            good=pred[list(models)[0]].notna()
            audit.append({'family':family,'horizon':horizon,'all_train_labels_known':bool((pred.loc[good,'train_end_max']<pred.loc[good,'asof_time']).all()),
                          'eval_sessions':len(ev),'forecast_coverage':float(ev[list(models)[0]].notna().mean()),
                          'observed_targets':int(ev.target.notna().sum()),
                          'ex_dividend_labels_excluded':int(excluded.loc[ev.index].sum())})
            # Descriptive development-only regression; never expose holdout for this diagnostic.
            valid_end=end<cutoff
            dx=panel[list(models.values())[-1] if family=='H1' else models['R3_skew_change']]
            tab=core.hac_ols(y.where(valid_end),dx.to_numpy(),list(dx.columns),max(5,horizon),positions=panel.session_id)
            tab['family']=family; tab['horizon']=horizon
            diagnostics.append(tab)
            if family=='H2':
                px=panel.loc[ev.index,'entry_price'].to_numpy()
                cash=panel.loc[ev.index,'entry_cash_dividend'].to_numpy() if 'entry_cash_dividend' in panel else None
                for model in list(models)+['buy_hold']:
                    prediction=np.ones(len(ev)) if model=='buy_hold' else ev[model].to_numpy()
                    if model=='buy_hold':
                        # Constant long exposure benchmark; one sleeve with no artificial ramp.
                        summ,stream=core.portfolio(prediction,px,1,args.cost_bps,cash)
                    else:
                        summ,stream=core.portfolio(prediction,px,horizon,args.cost_bps,cash)
                    stream.insert(0,'date',ev.date.iloc[:-1].to_numpy())
                    stream.to_csv(out/f'H2_h{horizon}_{model}_pnl.csv',index=False)
                    strategies.append({'horizon':horizon,'model':model,'cost_bps':args.cost_bps,**summ})
                for bps in sorted(set([0.,args.cost_bps*2,5.,10.])):
                    summ,_=core.portfolio(ev.R3_skew_change.to_numpy(),px,horizon,bps,cash)
                    strategies.append({'horizon':horizon,'model':'R3_cost_stress','cost_bps':bps,**summ})
                groups=forecast_groups(ev,'R3_skew_change')
                group_tables[f'H2_h{horizon}_forecast_groups']=groups
                if args.events and horizon==1:
                    counts,reg=event_analysis(panel,ev,read(args.events),meta['surface_kind'])
                    event_tables={'event_groups_exploratory':counts,'event_interactions_exploratory':reg}
    comparison=inference_families(pd.DataFrame(all_comparisons),meta['surface_kind'],not sensitivity)
    warnings=[
        'Availability times and source truth are supplied by the user; timestamp checks cannot authenticate the vendor.',
        'HAR = log-variance ridge HAR with training-only residual smearing; positive prediction, not an unbiased oracle.',
        'VRP_PROXY: implied variance (ATM squared unless model-free input supplied) minus 21-trading-day forecast. 30-calendar-day mismatch remains.',
        'Bootstrap is approximate and conditional on generated forecasts. It does not refit models or establish causality.',
        'Unique designated primary: curve H2 h=1 R3 vs R2, raw two-sided p. Auxiliary family gets separate Holm; this does not control the combined primary-plus-auxiliary family at 5%. All-comparison Holm is also reported as sensitivity.',
        'Inference families must be frozen before evaluating an untouched sample. Previously viewed results stay exploratory; software does not certify preregistration. Repeated tuning and event regressions are not covered.',
        'No Stambaugh correction is claimed. HAC/CW are descriptive; ridge does not remove persistence bias.',
        'Strategy diagnostics include supplied cash distributions; payment timing/tax/financing/borrow/roll and market impact beyond configured one-way cost are excluded. Missing dividend fields mean legacy price returns.',
        'An observed prediction does not identify insider information or dealer positioning.']
    if meta['availability_basis']!='verified':
        warnings.append('Availability basis is not verified; do not claim executable real-time performance.')
    if meta['surface_kind']=='proxies':
        warnings.append('Cboe proxy mode: SKEW-index change is NOT 25-delta skew change; original curve hypothesis is not tested.')
    if any(row['forecast_coverage']<.95 for row in audit):
        warnings.append('Forecast coverage <95% for some experiments; inspect missing-feature selection bias.')
    if coverage and min(coverage['complete_surface_fraction'],coverage['skew_change_fraction_excluding_first'])<.95:
        warnings.append('Development raw surface/skew-change coverage <95%, including pre-forecast history. No missing-node interpolation; inspect yearly build coverage and sample selection.')
    if exclude_ex:
        warnings.append('Ex-dividend exclusion is retrospective label conditioning, inclusive entry through exit. No calendar rows dropped; portfolio paths are NOT filtered using future ex-date knowledge.')
    if sensitivity:
        warnings.append('Sensitivity run: no primary hypothesis claimed here. Within-run Holm does not jointly adjust comparisons across variant runs.')
    input_hashes={k:filehash(getattr(args,k)) for k in ['input','calendar','metadata']}
    if args.events:
        input_hashes['events']=filehash(args.events)
    summary={'version':__version__,'phase':args.phase,'metadata':meta,'code_sha256':codehash(),'input_sha256':input_hashes,
             'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
             'config':{k:v for k,v in vars(args).items() if k!='func'},'holdout_start':raw.date.iloc[boundary],
             'feature_specs':{'H1':h1,'H2':h2},'development_surface_coverage':coverage,
             'study_role':'sensitivity' if sensitivity else 'designated_primary',
             'inference_families':{'primary_count':int((comparison.test_role=='primary').sum()),
                 'auxiliary_count':int((comparison.test_role=='auxiliary').sum()),
                 'registration_verified':False,'primary_p':'raw two-sided paired block bootstrap',
                 'auxiliary_p':'Holm within auxiliary comparisons in this run'},
             'audit':audit,'warnings':warnings}
    tables={'forecast_metrics':pd.DataFrame(all_metrics),'model_comparisons':comparison,
            'yearly_comparisons':pd.DataFrame(all_yearly),'strategy_diagnostics':pd.DataFrame(strategies),
            'development_HAC_descriptive':pd.concat(diagnostics,ignore_index=True),**group_tables,**event_tables}
    for name,table in tables.items():
        table.to_csv(out/(name+'.csv'),index=False)
    save_json(out/'summary.json',summary)
    make_html(out,summary,tables)
    print(json.dumps({'report':str((out/'report.html').resolve()),'phase':args.phase,
                      'synthetic':meta['synthetic'],'comparisons':len(comparison)},ensure_ascii=False,indent=2))


def plots(args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError('Optional plots need matplotlib: python -m pip install matplotlib') from exc
    directory=Path(args.report_dir)
    files=list(directory.glob('H2_h*_R3_skew_change_pnl.csv'))
    if not files:
        raise ValueError('No H2 PnL files found.')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for f in files:
        d=pd.read_csv(f)
        label=f.name.split('_')[1]
        axes[0].plot(pd.to_datetime(d.date),d.wealth,label=label)
        peak=np.maximum.accumulate(np.r_[1.,d.wealth.to_numpy()])[1:]
        axes[1].plot(pd.to_datetime(d.date),d.wealth/peak-1,label=label)
    axes[0].set_title('R3 diagnostic wealth (not execution performance)'); axes[1].set_title('Drawdown')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(directory/'diagnostic.png',dpi=160)
    plt.close(fig)
