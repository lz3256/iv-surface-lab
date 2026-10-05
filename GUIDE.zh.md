# IV Surface Lab · 统一研究版 v3.2

这是对之前 Codex 的 `iv_surface_lab` 和用户提供的 Claude `ivsurf_v2` 的实质整合：**一个数据格式、一套时间审计、一个命令同时跑收益与波动率研究**。HAR已进入模型，VRP有单独检验，GEX保留为探索模块。原两个项目保持不变。

它是完整的研究工具，不是已经证明有alpha的交易系统。交付报告和示例均为模拟数据。

**v3.2 增量更新**：`build-wrds`增加`ret/retx`现金分配推导，H2和损益计入应得分红，隔夜方差同步调整；区分实际退出时间与标签可知时间。新增50Δ call/put差异诊断、早期曲面及差分覆盖率，预先指定扩展窗口/除息剔除变体。此前三表构建、真实XNYS日历和主/辅助检验家族保持。直接接数据先读 [WRDS_BUILD.md](WRDS_BUILD.md)，冻结规则见 [EXPERIMENT.md](EXPERIMENT.md)。压缩包仅含源码、测试、文档和小模板；模拟数据与报告用 `run_demo.py` 重新生成。

## 1. 最快运行方式

### 当前这台 Mac：不联网、不安装依赖

```bash
cd <local-project-path>
python3 run_demo.py
```

程序会创建一个带时间标记的 `runs/demo_.../` 目录，完成：

1. 生成模拟日历、市场数据、曲面、事件和期权链。
2. 将长表曲面导出为统一CSV，执行时间与日历验证。
3. 运行HAR、H1、H2、事件分组、成本与统计检验。
4. 运行探索性GEX。
5. 打印可用浏览器打开的 `research/report.html` 路径。

报告目录不允许覆盖，保留每次实验记录。模拟数据带有人工植入信号，不代表市场收益。

### 其他机器：Python 3.10+

```bash
cd ivsurf_complete
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
```

Windows激活命令为 `.venv\Scripts\activate`。核心运行只依赖 NumPy、pandas。WRDS下载和PNG图表为可选依赖；HTML报告无需联网或绘图库。

## 2. 合并了什么

| 功能 | 统一版实现 |
|---|---|
| 数据入口 | 标准化曲面长表、已有IV/Delta的报价、Cboe指数CSV；build-wrds在线三表/本地三文件构建 |
| 时间审计 | 明确数据可得/价格观察/成交时间，按真实标签结束时间截断训练 |
| 交易日历 | 独立日历对照；build-wrds用exchange_calendars XNYS生成含半日/夏令时的日历 |
| 期限 | 曲面统一30/91日；Cboe VIX3M保持3M名称，不冒充91日 |
| 实际方差 | Garman–Klass日内方差代理＋隔夜对数收益平方 |
| HAR | 对数方差HAR ridge，训练残差smearing转换回正的方差预测 |
| H1 | 未来5/21日实际方差：HAR、加隐含方差、再加曲面形状 |
| H2 | 未来1/5/21交易时段收益：市场控制、加HAR、再加偏斜变化 |
| VRP | 明确标记代理，单独检验；不与其两项组成部分一起造成精确共线 |
| 统计 | 样本外误差、块bootstrap；1项指定主检验原始p＋辅助Holm，附全部比较Holm敏感性、描述性HAC/CW |
| 交易诊断 | 多日等权交错子组合，逐日收益、换仓、最终平仓与成本敏感性 |
| 事件 | 1日收益窗口内/入场前事件分组及描述性交互回归 |
| GEX | 精确剩余时间、利率/分红、实际持仓或显式OI假设、价格网格gamma flip |
| 审计 | 代码/输入哈希、参数、每次训练的标签最大结束时间、封存区间、运行版本 |

没有照搬旧代码的Stambaugh修正。原实现的对齐和标准误解释尚不足以作为可信模块；这里使用清楚标注的HAC/预测误差推断，**不声称ridge消除了Stambaugh偏差**。也没有把全样本GEX分组包装成样本外交易结果。

## 3. 最少需要四类真实输入

### A. 日历 `calendar.csv`

| 列 | 含义 |
|---|---|
| `date` | YYYY-MM-DD，交易所session标签 |
| `session_open` | 带时区的实际开市时间 |
| `session_close` | 带时区的实际收市时间，包含提前收市 |

手动export时，需提供所选市场可信的完整交易所日历，至少覆盖全部信号日期和入场日期；build-wrds会生成XNYS日历并记录包版本。校验器会核对CSV是否漏日，但不能独立认证任意用户日历。不要把普通周一至周五日历当成真实日历。演示日历只有周末过滤，已明确标为模拟。

### B. 市场与时点 `market.csv`

| 列 | 含义 |
|---|---|
| `date` | 信号对应session，与日历一致 |
| `reference_time`, `reference_price` | 实际观察的参考价格和观察时间，用于过去收益特征 |
| `asof_time` | 信号计算时，全部特征已知的时点 |
| `surface_available_time` | 曲面实际可得时间或明确声明的假设时点 |
| `rv_available_time` | 当日完整OHLC/RV可得时点，不得早于session收市 |
| `open`, `high`, `low`, `close` | 同一口径的日OHLC，用于方差估计 |
| `entry_session` | 实际入场session标签，例如次日 |
| `entry_time`, `entry_price` | 匹配同一真实时间的可执行标的价格 |
| `cash_dividend`, `is_ex_dividend` | 分红调整输入的当日现金分配和标记，供隔夜方差使用 |
| `entry_cash_dividend`, `entry_is_ex_dividend`, `entry_dividend_available_time` | 对应入场日的事后分配金额、标记和可知时间；只作标签/诊断，不进入特征 |

分红列为一整组，与`metadata.return_basis=cash_distribution_total_return`配合；缺一列或缺金额会报错。旧模板不含这组字段，仍表示价格收益；WRDS构建器自动生成新口径。`data/templates/execution_wrds.csv`提供新CRSP原始字段表头。

所有时间必须有时区，例如 `2024-01-03T09:31:00-05:00`。代码要求：

```text
reference_time <= asof_time
rv_available_time <= asof_time < entry_time
surface_available_time <= asof_time
session_open(entry_session) <= entry_time <= session_close(entry_session)
```

**不再默认“18:00已收到供应商文件”。** 若不能确认可得时间，在metadata中声明`assumed`；结果只能视为情景研究，不能声称真实可交易。

**不再把开盘价标成09:31。** 如果用09:31执行，就提供09:31价格；若只有正式开盘价，填匹配的实际开盘时点并明确开盘成交假设和成本。程序无法鉴别用户手动填写的价格是真是假，因此仍需对源数据抽样核对。

**不允许缺成交价后删行。** 缺失市场价格会报错；缺失曲面可以留NA，整个交易日必须保留。跨日收益标签和持仓长度都由连续session构造。

当前为**日终信息模型**：必须等当日OHLC已知。它不是盘中HAR/5分钟研究框架。要做盘中预测，需另行定义特征截止时点与已实现波动区间，不能复用当天完整OHLC。

### C. 曲面 `surface_long.csv` 或已合并的 `daily.csv`

长表字段：

```text
date, days, delta, cp_flag, impl_volatility
```

需要30和91日的50-delta call，以及30日25-delta put/call。IV是年化小数，例如 `.20`；`cp_flag`为C/P。delta默认百分数单位，call50、call25、put-25；小数单位需传 `--delta-unit decimal`。

标准化后四列：

```text
iv_atm_30, iv_atm_91, iv_put25_30, iv_call25_30
```

`iv_atm_*`在本项目中是**50-delta call的平值代理**，不是严格远期ATM。对供应商曲面须保持定义一致。91日不重命名成90日。

若已有以上四列，可与market字段一起提供 `daily.csv`，跳过export。

可选提供 `implied_var_30`（年化小数方差），替代ATM IV平方作为隐含方差输入。必须在metadata中记录构造方法；代码不会自动保证它是无模型方差。

### D. 元数据 `metadata.json`

模板在 `data/templates/metadata.json`。必须填写：

```json
{
  "synthetic": false,
  "surface_kind": "curve",
  "availability_basis": "assumed",
  "reference_instrument": "SPX",
  "execution_instrument": "SPY",
  "price_basis": "请明确分红、复权、换月与价格来源",
  "iv_level_definition": "50-delta call IV proxy; 30/91 calendar days",
  "execution_price_time_basis": "请明确执行时间与价格的对应关系",
  "calendar_source": "请填写实际交易所日历来源",
  "availability_evidence": "请填写供应商交付证据，或说明这是时点假设"
}
```

`availability_basis`只能是`verified`、`assumed`或`synthetic`；填写verified不是程序替你验证了事实。

SPX曲面搭配SPY或ES是跨资产映射，需要记录和验证。不要把SPX指数价视为可执行价。期货不同月份合约不能直接拼价格求收益；ETF除息、复权、借券与融资也需要单独明确。当前收益和交易诊断基于你提供的价格路径，不自动补这些项。

## 4. 接入真实数据

输入模板均在 `data/templates/`。

### 标准化曲面导出与检查

```bash
python -m ivsurf export \
  --surface data/real/surface_long.csv \
  --market data/real/market.csv \
  --calendar data/real/calendar.csv \
  --metadata data/real/metadata.json \
  --out data/real/daily.csv

python -m ivsurf validate \
  --input data/real/daily.csv \
  --calendar data/real/calendar.csv \
  --metadata data/real/metadata.json
```

市场时间全部由输入提供；export不创造可得时间，不改变成交价对应时间，不删除交易日。会生成manifest记录输入哈希。

### WRDS三表构建（推荐，需账户与权限）

`build-wrds` 已接入命令行。它查询 `vsurfd`、`secprd` 和 CRSP 日表，输出 `daily.csv/calendar.csv/metadata.json`、原始快照与质量记录。也能离线接收三份原始CSV。具体字段、安装和完整命令见 [WRDS_BUILD.md](WRDS_BUILD.md)。

```bash
python -m pip install -r requirements-wrds.txt
python -m ivsurf build-wrds \
  --option-library optionm --secid 108105 \
  --crsp-library crsp --spy-permno 84398 \
  --start 2010-01-01 --end 2025-12-31 --out data/real/build_01
```

以上库名、ID均须在账户核实。默认：18:00假设曲面可得；实际XNYS收盘＋5分钟假设日价格可得；次日计划开盘（通常09:30）按CRSP开盘价假设成交。`availability_basis=assumed`，不是实时认证。secprd仅取SPX参考收盘；OHLC/RV取SPY并修正除息隔夜跳空，跨标的基差仍存在。不要将开盘价记为09:31。v3.2本地CRSP输入需要ret/retx，并比信号区间多取前一交易日和后一交易日。

### 仅下载WRDS曲面（保留旧命令）

```bash
python -m pip install -r requirements-wrds.txt
python -m ivsurf wrds-download \
  --option-library optionm \
  --secid 108105 \
  --start 2010-01-01 --end 2025-12-31 \
  --out data/raw_wrds
```

库名和secid只是沿用此前项目的**示例参数，必须核实账户实际schema和证券标识**。下载前程序检查年度表及字段存在，查询值参数化；不会猜另一个库替代。连接由WRDS客户端处理，不把密码写入项目。

`wrds-download`旧命令仍只下载曲面；要自动生成日历及假设时间输入用上面的`build-wrds`。两个命令均不提供经验证的供应商发布时间、分钟成交价或真实dealer持仓，也未在你的WRDS账户上实测；本地测试使用mock连接验证参数化、字段检查和文件生成。

来源：[WRDS官方Python实现](https://github.com/wharton/wrds/blob/main/wrds/sql.py)、[OptionMetrics产品说明](https://optionmetrics.com/data-products/)。

### 原始报价带IV/Delta

```bash
python -m ivsurf quotes \
  --quotes data/real/quotes.csv --market data/real/market.csv \
  --calendar data/real/calendar.csv --metadata data/real/metadata.json \
  --out data/real/from_quotes.csv
```

报价字段见模板。`contract_id`用于每个合约只取`reference_time`前最后一条报价；使用实际到期时间。默认报价最多陈旧120秒，相对价差≤50%，bid>0，剩余7—200日。先按delta插值，再按总方差跨期限插值，不外推、不前向填充，另存质量CSV。

先自行筛选单一标的和一致的合约规则。此模块依赖供应商已有IV/Delta，**不是完整无套利曲面拟合或原始价格反解器**。

### Cboe免费指数代理路线

从[Cboe历史数据页](https://www.cboe.com/tradable_products/vix/vix_historical_data)取得 `VIX.csv`、`VIX3M.csv`、`SKEW.csv`、`VVIX.csv`，放同一目录；文件需date和close或指数同名值列。`surface_kind`改成`proxies`：

```bash
python -m ivsurf cboe \
  --directory data/cboe --market data/real/market.csv \
  --calendar data/real/calendar.csv --metadata data/real/proxy_metadata.json \
  --out data/real/proxy_daily.csv
```

程序不抓取实时网页、不替你下载标的价格。代理路线用SKEW指数变化，**不能声称检验了25-delta曲面变化**；报告会明确标注。VIX3M保留3M定义，不命名成91天。

## 5. 统一研究命令

```bash
python -m ivsurf run \
  --input data/real/daily.csv \
  --calendar data/real/calendar.csv \
  --metadata data/real/metadata.json \
  --out reports/real_dev_001 \
  --phase development
```

默认：HAR最少252个有效训练样本、主模型最少504个、训练窗口最多1008个、每21行重拟合、ridge alpha=10、最后252行保留。嵌套HAR会增加热身期；建议先准备覆盖多个市场阶段的较长历史，演示为2200行。训练行数不是统计功效保证。

主参数可用 `python -m ivsurf run --help` 查看。所有参数写入summary.json。

### H2模型含义（主线）

| 模型 | 特征 |
|---|---|
| R0_mean | 当前训练集历史平均收益 |
| R1_market | 1/5日参考价格收益、过去22日实际方差、隐含方差、期限斜率 |
| R2_HAR | R1＋HAR预测的未来21交易日平均方差 |
| R3_skew_change | R2＋30日期权偏斜的单日变化 |
| R4_VRP_only | 只用VRP代理的辅助实验 |

主比较是 **R3 vs R2、1日收益**。5日/21日为辅助检验。每个期限的模型共享训练样本，以避免各自删缺失值改变对照。

新版WRDS标签使用`(退出开盘价＋入场日之后至退出日的现金分配之和)/入场开盘价−1`。除息日开盘买入不赚当天分红；跨到除息日开盘退出会计入。多日现金不再投资。分红由当日RET/RETX推导，因此标签可知时间保守延迟至退出日收盘后，不能当成入场时已知的公告。`target_exit_time`与`target_end`分别表示实际退出和标签可知；旧无分红输入保留原价格收益行为。

协议已指定`--window 1000000 --study-role sensitivity`扩展窗口和`--exclude-ex-dividend --study-role sensitivity`回顾性除息剔除。后一选项只屏蔽训练/评估标签，不删除日历行、不利用未来分红日期过滤策略交易。两个变体都不能被选作新的主检验；跨运行的辅助校正范围见EXPERIMENT.md，软件没有自动汇总跨运行p值。

`VRP_proxy = iv_var_30 - har_fwd_21`。R2已含这两项，再加VRP会精确线性依赖，所以不重复添加。30日历日与21交易日只是近似匹配；ATM IV²也不是无模型方差。这里始终标为代理。

### H1模型含义（副线）

预测未来5或21交易日的年化平均实际方差，回归目标取log：

- V0_mean：历史平均log方差。
- V1_HAR：过去1/5/22日log实际方差。
- V2_IV：再加log隐含方差。
- V3_surface：再加期限斜率、偏斜、曲率（代理路线使用相应指数）。

主比较是 **V3 vs V2**，避免把“IV水平有用”误认为“整个曲面形状有用”。H1误差在log方差尺度评价，不能与H2收益误差直接比较。

### 封存测试

```bash
python -m ivsurf run \
  --input data/real/daily.csv --calendar data/real/calendar.csv \
  --metadata data/real/metadata.json --out reports/real_final_001 \
  --phase final
```

须保持开发时冻结的参数。开发阶段先截去封存数据再构造特征/标签，不能产生跨入封存期的训练结果。final按事先设定规则逐次吸收已实现的收益，这属于顺序预测，不是冻结整个模型。看过final结果后据此改参数，就需要新的未见样本。

## 6. 事件和GEX

### 事件（有日历才运行）

提供`event_id,event_type,event_time`，时间含时区，event_id唯一；这是当时已知的计划公告日历，不是事后挑出的波动日。

```bash
python -m ivsurf run ... --events data/real/events.csv
```

省略号代表上一节完整的输入参数，不能原样粘贴。模块只在1日收益实验中区分：

- 入场后、退出前的公告。
- 信号已形成但尚未入场时的公告。
- 两者都没有。

例如CPI在08:30发布而09:31才入场，不能把08:30的涨跌计成可交易领先收益。输出实际事件数、分组均值、描述性交互回归。当前没有宏观surprise预测、精确公告分钟窗口或因果识别；事件回归不纳入主预测比较的Holm家族，应视为探索结果。

### GEX（独立探索模块）

```bash
python -m ivsurf gex \
  --chain data/real/chain.csv --daily data/real/daily.csv \
  --calendar data/real/calendar.csv --metadata data/real/metadata.json \
  --position-mode heuristic --out reports/gex_001
```

链字段见模板。需要实际`snapshot_time`、`available_time`、`expiry_time`，标的价、执行价、IV、利率、分红收益率、合约乘数和OI。利率与分红收益率为年化连续复利小数。

- `heuristic`：显式假设做市商long call/short put，绝非真实持仓测量。
- `dealer`：需额外提供有符号的`dealer_net_contracts`；其数据来源仍需自行核实。

不再用期限结构冒充GEX。0DTE使用实际剩余秒数；已到期合约排除，不把到期日午夜差值强行设成极小正数。假设IV按执行价固定，改变标的价格重算总gamma，在网格内找根；多根或无根均记录。高于最近穿零点不自动代表正gamma。

该模块只分析开发区间，默认同样保留252行。输出为描述性敞口和回归，未开展GEX样本外策略。

## 7. 输出、统计和风险边界

报告为离线HTML，全部结果也有CSV：

| 文件 | 用途 |
|---|---|
| `summary.json` | 参数、版本、哈希、封存日期、数据口径、警告 |
| `features.csv` | 包含HAR预测和VRP代理的特征 |
| `har_audit.csv` | 每日HAR预测及训练标签最大结束时间 |
| `H1/H2_h*_predictions.csv` | 真实目标、模型预测、训练量、训练截止时间；H2另含实际退出时间与除息屏蔽标记 |
| `model_comparisons.csv` | 增量R²、误差改善区间、原始p、test_role、p_primary、p_holm_aux、p_holm_all_sensitivity |
| `yearly_comparisons.csv` | 主比较的分年稳定性 |
| `H2_h*_forecast_groups.csv` | 用过去预测值决定的分组，不用全样本分位数 |
| `H2_h*_*_pnl.csv` | 每日仓位、收益、成本、财富路径 |
| `strategy_diagnostics.csv` | 仓位暴露、成本敏感性、回撤等 |
| `development_HAC_descriptive.csv` | 开发数据描述性回归，不是OOS证据 |
| `event_*` | 可选事件分组与描述性回归 |

bootstrap对成对预测损失序列做循环时间块抽样，保留缺失位置，每次重算有效样本均值；默认块长至少达到预测期限。它不重新拟合模型，也不解决所有非平稳性。p值为中心化bootstrap的近似双侧检验。真实曲面H2 h=1 R3对R2是唯一指定主比较，报告原始p；默认其他12项构成辅助Holm家族。另保留全部13项Holm作为敏感性；代理模式没有原主比较，全部归辅助。该分组不控制两家族合计的整体错误率，也不能抵消跨运行反复选模型。必须在未见样本上事前固定协议，已看过的结果仍是探索性的。

HAC考虑原session间距；Clark–West使用至少与期限相当的带宽，但在ridge/复杂估计设定下仅作为描述性诊断，不直接当作具有标准临界值的正式检验。没有声称使用了Stambaugh偏差修正。

多日收益策略通过每日新增1/h权重、持有h个时段的交错信号形成目标仓位，逐日重平衡/盯市近似；绝不把重叠21日收益直接年化成日频Sharpe。费用按净目标仓位变化计算，首笔开仓与最终平仓均收费。提供分红字段时计入多头应得/空头应付现金，按日复合收益；这与多日标签持有现金不再投资的规则不同。仍是固定名义敞口诊断，未精确计入价格漂移再平衡成本、分红支付延迟/税、换月、借券、融资、冲击和容量；现金利息为零。

相同波动率/风险预算的策略对照、执行延迟敏感性、真实市场成本、独立事件样本功效仍需用真实数据补充。预测改善不等于因果关系，也不能识别内幕消息。

## 8. 测试与复现

```bash
python -m unittest discover -s tests -v
```

测试覆盖：日历缺口、缺价格、执行时间、数据可得时间、隔夜方差、多期标签、未来扰动不改变过去预测、HAR时间审计、91日期限映射、报价过滤、成本/多日子组合、Holm、HAC、GEX、mock WRDS和完整开发/最终流程。

零信号对照：

```bash
python -m ivsurf demo --out data/null_check --signal-strength 0
python -m ivsurf run \
  --input data/null_check/daily.csv --calendar data/null_check/calendar.csv \
  --metadata data/null_check/metadata.json --out reports/null_check
```

单个随机种子的零信号实验不是长期假阳性率校准，不应据此宣传模型有效。

可选静态图（不影响HTML报告）：

```bash
python -m pip install -r requirements-plots.txt
python -m ivsurf plots --report-dir reports/real_dev_001
```

## 9. 文件导航

```text
run_demo.py               一键离线完整示例
ivsurf/__main__.py        统一命令行
ivsurf/core.py            校验、特征、HAR、滚动预测、统计、持仓
ivsurf/data.py            标准曲面/报价/Cboe/WRDS入口、模拟数据
ivsurf/research.py        H1/H2、事件、汇总报告、可选图表
ivsurf/gex.py             单独探索性gamma模块
tests/test_complete.py    自动验证
data/templates/          真实输入模板
MERGE_NOTES.md            两份项目如何合并、修复与保留限制
EXPERIMENT.md             真实实验前冻结的协议
VALIDATION.md             本次实际运行记录
```
