# 下一步：冻结协议并核验20天样本

这是一份执行清单，不是已经完成的预注册。代码仍为v3.2；没有读取真实账户或样本。当前项目目录尚未初始化Git。

## 1. 确认数据权限

登录自己的WRDS账户，确认可以查询这两类数据：

- OptionMetrics IvyDB US：SPX标准化波动曲面、标的价格。
- CRSP：SPY每日价格，含openprc、prc、cfacpr、ret、retx。

只需反馈“都有 / 权限不确定 / 没有”，不用分享账号、密码、验证码。若没有相应许可，先解决学校/机构的数据访问；不要把Cboe SKEW代理当作原25Δ曲面实验。

## 2. 填协议，而不是自行猜供应商事实

打开EXPERIMENT.md。下面可作为候选填写内容；正式冻结前确认接受这些选择，并填完账户实际信息。

| 空白 | 候选填写内容 |
|---|---|
| 市场/合约族 | 美国SPX指数期权标准化30/91天曲面；30天25Δ put/call偏斜变化 |
| 参考/执行/RV | SPX参考收盘；SPY次日开盘成交假设；SPY分红调整OHLC方差代理 |
| 日历 | exchange_calendars XNYS；记录你实际安装的版本；保留假日、半日及夏令时 |
| 数据来源/版本/授权 | **必须填实际情况**：WRDS库/表、数据版本（若供应商未提供版本号就明确写未提供）、提取日期和机构访问权限；确认SPX secid、SPY permno |
| 可得时间 | 未取得历史发布记录；surface纽约18:00、日价格收盘+5分钟均为assumed；asof为较晚者+1分钟 |
| 执行/费用 | 次日官方开盘价假设成交；单边1bp作为研究基准，不是实测交易费用；现有诊断另报0/2/5/10bp |
| 分红/公司行动 | RET/RETX推导现金等价值，抽样核验实际分配；跨除息日计入，入场日分红不计；拒绝拆股因子变化；支付延迟/税/融资/借券未建模 |
| 候选全样本 | 信号日期2006-01-01至2025-12-31；有效首日2006-01-03。**这是建议的截止日期，尚未被你确认。** |
| 候选开发/确认区间 | 在上述区间和XNYS完整日历下，共5031行；前4779行截至2024-12-27为开发期，最后252行2024-12-30至2025-12-31为确认期 |
| 主参数 | 使用当前默认：HAR min252、主模型min504、window1008、refit21、alpha10、block20、bootstrap1000、holdout252 |
| 新增敏感性 | 只做协议已明确的扩展窗口和除息剔除；不增加二者组合 |
| 尚未定下的网格/多市场/事件 | 第一版不做额外报价过滤网格、块长网格、正则化搜索、多市场或确认性事件检验；已有事件/GEX模块不纳入本次真实主实验 |
| 覆盖率 | 95%为报告警告阈值；不插值、不据结果改样本起点，低覆盖率记录原因和样本损失 |
| ES部分 | 当前未实施、未获取CME数据，其细节标为本次不适用；将来另立协议，不留作本次随意增选项 |

如果选择其他截止日期，最后252行的起点必须重新根据实际日历计算，不能照抄上表。信号行完整，曲面缺失不删日；最终252行的尾部会因没有未来退出标签而少于252个可评分目标，尤其21日期限。报告应写实际n。

核验真实数据的机构授权、表字段和样本日期是数据工程，不是看回测收益。主实验规则要在首次真实开发回测前冻结；冻结后因字段错误而修订，记录原因和新提交，不覆盖旧历史。

## 3. 在独立项目目录保存协议和代码版本

先填完上面的真实信息，才运行以下命令。它们只创建本地版本记录，不会上传数据或代码。

```bash
cd iv-surface-lab
git init
git add EXPERIMENT.md ivsurf tests requirements*.txt README.md WRDS_BUILD.md START_HERE.md .gitignore
git commit -m "Freeze IV surface protocol before real-data development"
git rev-parse HEAD
```

记下最后的提交哈希。提交本身不证明实验从未被看过；registration_verified仍为false。如Git提示没有提交者身份，使用自己的实际身份配置本仓库，不能填写编造的身份。数据和账户凭据不要提交。

## 4. 准备环境

已有可用的v3.2环境就继续用，不必重装。新环境可以这样创建：

```bash
cd iv-surface-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-wrds.txt
mkdir -p data/raw/sample_20
```

本地CSV模式仅需requirements-calendar.txt；在线WRDS和Parquet才需要对应额外依赖。

## 5. 在WRDS导出这三个CSV

建议字段核验区间：**2024-09-03至2024-09-30，20个XNYS交易日**。CRSP多取前后各一个交易日：**2024-08-30至2024-10-01，22行**。日期数由exchange_calendars 4.13.2核对；不是已经拿到数据的行数。

| 保存文件 | 筛选 | 必须导出的列 |
|---|---|---|
| data/raw/sample_20/surface_long.csv | 已确认的SPX；2024-09-03至09-30；days为30/91；两侧均保留25Δ、50Δ节点 | secid,date,days,delta,cp_flag,impl_volatility |
| data/raw/sample_20/secprd.csv | 同一SPX；2024-09-03至09-30 | secid,date,close |
| data/raw/sample_20/crsp_daily.csv | 已确认的SPY；2024-08-30至10-01 | permno,date,openprc,askhi,bidlo,prc,cfacpr,ret,retx |

两侧50Δ都保留，才能计算ATM差异诊断；仅保留模型必需四个节点会让该诊断缺少put。可直接导出30/91天的所有delta，由构建器选择，不需要手动拼每日四列。

日期使用YYYY-MM-DD；IV为年化小数，默认delta百分数、put负数。不要前填缺失、不把负价格取绝对值、不手工算分红、不自行造时间列。真实库/表命名可能随许可版本不同；若是CIZ新字段，先核对映射，不能随便改一个列名冒充旧口径。

核对当月分配记录；如果样本中有现金分配，应在manifest中看到相应事件。不凭预期把dividend金额填成非零。

## 6. 构建和校验，只核数据

下面108105、84398仍是项目示例ID，**只有在账户确认分别对应SPX/SPY后才能原样用**，否则替换为已核实值。

```bash
python -m ivsurf build-wrds \
  --surface-file data/raw/sample_20/surface_long.csv \
  --reference-file data/raw/sample_20/secprd.csv \
  --execution-file data/raw/sample_20/crsp_daily.csv \
  --secid 108105 --spy-permno 84398 \
  --start 2024-09-03 --end 2024-09-30 \
  --out data/real/sample_20_v1

python -m ivsurf validate \
  --input data/real/sample_20_v1/daily.csv \
  --calendar data/real/sample_20_v1/calendar.csv \
  --metadata data/real/sample_20_v1/metadata.json
```

成功时应有20个信号行；availability_basis应为assumed。检查:

- quality.atm_call_put_diagnostic：配对覆盖率；缺put显示null，不是“没有差异”。
- quality.dividends：RETX价格恒等式误差、现金事件数，与原始记录核对。
- quality.coverage_by_year及quality.skew_change_coverage_excluding_first：节点和差分覆盖率分别看。

报错时保留原始文件和错误文本；不要为了通过validate删交易日。修正后使用新的输出目录，例如sample_20_v2，保留版本对照。

## 7. 发回什么

这一步完成后，提供：

1. 协议提交哈希，以及仍未能核实的假设。
2. validate的完整终端输出，或报错原文。
3. build_manifest.json。必要时再提供脱敏的列名和少量记录；不要提供账户凭据。

20天样本不运行统计模型，不需要尝试缩小min-train来让run成功。字段、现金分配和日期覆盖核对通过后，再取全样本；到那一步再执行一次开发主线和两个指定敏感性。

最终阶段只运行冻结的主线。真正会消耗确认样本的是查看并据此使用结果，而不是机器命令的执行次数：纯技术失败、尚未查看任何结果的重试要完整记录；已经看过结果，就不能通过换目录重跑把它恢复为未见样本。
