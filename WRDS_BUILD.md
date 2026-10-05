# 从 WRDS 到可运行输入：v3.2

`build-wrds` 支持在线查询三张表，也支持读取三个原始 CSV/Parquet。v3.2增加现金分配调整、ATM两侧定义诊断及差分后覆盖率。20 天样本足以核对字段和时点，但不足以做默认 HAR + H2 实验。

## 安装

```bash
cd ivsurf_complete
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-wrds.txt
python -m unittest discover -s tests -v
```

只有本地 CSV 时可改装 `requirements-calendar.txt`；Parquet 另需 `pyarrow`。核心模拟实验仍只需 `requirements.txt`。

## 在线构建

以下库名、表模式、证券 ID 是示例，先在你的 WRDS 账户确认。尤其要确认 `secid` 为 SPX，`permno` 为 SPY；程序核查字段和 ID 一致性，不会凭 ID 数字认证证券名称。

```bash
python -m ivsurf build-wrds \
  --option-library optionm --secid 108105 \
  --surface-table 'vsurfd{year}' --reference-table 'secprd{year}' \
  --crsp-library crsp --crsp-table dsf --spy-permno 84398 \
  --start 2010-01-01 --end 2025-12-31 \
  --assumed-surface-time 18:00 --assumed-price-delay 5 \
  --out data/real/build_01
```

如果账户提供的是不分年的 `secprd`，明确指定 `--reference-table secprd`；其他表也支持固定名称或 `{year}`。表/列不存在会报错，不能静默改用另一个可能不同口径的字段。WRDS 客户端负责认证；不要把账号密码写进代码或提交到 git。

| 原始表 | 必需列 | 用途 |
|---|---|---|
| OptionMetrics `vsurfd{year}` | `secid,date,days,delta,cp_flag,impl_volatility` | 30/91天50Δ call、30天25Δ put/call |
| OptionMetrics `secprd{year}` 或账户的 `secprd` | `secid,date,close` | SPX 参考收盘价及历史收益控制 |
| CRSP 日表，例如 `dsf` | `permno,date,openprc,askhi,bidlo,prc,cfacpr,ret,retx` | SPY OHLC、开盘成交价及现金分配推导 |

IvyDB 的 Security Price 文件**确实定义了开、高、低、收字段**；其中开盘价不可得时可为0。选择SPY OHLC的理由是测量口径：SPX由成分股计算，成分股开盘不同步，指数日内开盘与等待各股开盘的特别开盘结算值（SOQ）也不是一回事。陈旧/不同步价格可能影响隔夜和区间方差，因此这里用可交易ETF自身的OHLC，与执行标的一致；不声称偏差在所有情形都必然向下，也不声称SPY日线等同高频真实方差。[IvyDB官方手册，Security Price](https://lib.tsinghua.edu.cn/PDF/OptionMetricsshiyongzhinan.pdf)、[S&P指数数学方法，SOQ](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-index-math.pdf)。

H1/HAR仍是SPY Garman–Klass＋分红调整后隔夜平方收益代理，不是SPX 5分钟RV。SPX曲面与SPY方差的基差仍影响VRP解释。未来有SPX/ES日内数据后应另设实验，不能悄悄换目标定义。

默认 `delta` 为百分数，put 为负：`-25`，call 为 `25/50`；若原始导出是小数，用 `--delta-unit decimal`。不要凭结果改符号。IV 必须是年化小数。

## 已有 20 天原始文件

文件使用上表的列名。OptionMetrics覆盖所有信号日；CRSP必须额外包含**首个信号日前一交易日**（现金分配推导基准收盘）和**最后信号日之后的下一个交易日**（最后一次入场）。例如下方11月1日至29日样本，CRSP要覆盖10月31日至12月2日。曲面节点允许缺失，市场价格不能缺行；现金分配不能把缺失当零。

```bash
python -m ivsurf build-wrds \
  --surface-file data/raw/surface_long.csv \
  --reference-file data/raw/secprd.csv \
  --execution-file data/raw/crsp_daily.csv \
  --secid 108105 --spy-permno 84398 \
  --start 2024-11-01 --end 2024-11-29 \
  --out data/real/sample_check

python -m ivsurf validate \
  --input data/real/sample_check/daily.csv \
  --calendar data/real/sample_check/calendar.csv \
  --metadata data/real/sample_check/metadata.json
```

这条路径不连接 WRDS。三份文件必须一起提供；保留原始字段，不要先删掉缺曲面的日期。

## 时间与价格到底如何对应

| 项目 | 默认规则 | 证据状态 |
|---|---|---|
| 日历 | `exchange_calendars` 的 XNYS，显式指定历史范围 | 独立日历；记录包版本 |
| `reference_time` | t 日 XNYS 实际计划收盘 | SPX 日线收盘的日频映射假设；不认证期权快照同步性 |
| `rv_available_time` | 该收盘时间＋5分钟 | 假设价格已可得；不是 WRDS 历史发布记录 |
| `surface_available_time` | t 日纽约18:00 | 假设；不保证历史文件当时已发布或未被修订 |
| `asof_time` | 上述两种可得时间较晚者＋1分钟 | 计算延迟假设 |
| `entry_time` | t+1 交易日计划开盘，通常纽约09:30 | 按 CRSP 日开盘价成交的假设 |
| `entry_price` | t+1 的 `openprc` | 日线没有提供实际开盘打印/个人成交时间 |
| `entry_dividend_available_time` | 入场日实际收盘＋5分钟 | 同一行的入场日分红是事后标签资料，不能放入特征 |

全部输出时间包含时区。日历会跳过假日，保留13:00提前收市和夏令时。**09:30 开盘价不能标成09:31价格**。需要09:31或18:30执行时，必须另取匹配时刻的数据。

`availability_basis` 始终为 `assumed`；本命令没有开关能将这些假设冒充为 `verified`。即便所有检查通过，也只能研究该时间假设下的滞后预测，不能声称验证了实时可交易性。

负 `prc` 可能表示买卖报价均值，本构建器拒绝 `abs(prc)` 自动修补。OHLC非正、缺价格、重复日期或证券混杂会报错。`cfacpr`在前置日、样本及末次入场日期间变化时也会拒绝：先处理公司行动，避免把拆股当收益。

## 分红口径与不能提前知道的字段

不假定所有版本的`dsf`都有`divamt`；构建器明确要求日线`ret`与`retx`。在价格/股数调整因子不变的当前SPY限定路径中，推导：

```text
D_t = C_(t-1) × (ret_t − retx_t)
核验：retx_t ≈ C_t / C_(t-1) − 1    （绝对容差 10^-6）
```

这是从含分配/不含分配的日收益恒等式推导的**现金等价值**，不替代对实际分配类型的核验。特殊分配、分拆和不同复权口径不能直接套用。`cfacpr`变化、缺失`ret/retx`、恒等式不符或负现金等价值会拒绝；金额绝对值不超过10^-6美元按数值零处理并记入审计。真实20天样本最好包含一个除息日，核对实际现金分配记录。CRSP新版CIZ若字段不同，先做显式字段映射，不能静默混用。[CRSP计算方法](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-calculations-index-methodologies-guide-flat-file-format-2-0/)、[WRDS对新版/旧版分配再投资差异的说明](https://wrds-www.wharton.upenn.edu/documents/2084/Webinar.pdf)。

`market.csv`和`daily.csv`新增当日`cash_dividend/is_ex_dividend`，以及对应入场日的`entry_cash_dividend/entry_is_ex_dividend/entry_dividend_available_time`。金额和标记必须完整一致。

设入场开盘为O_e，退出开盘为O_x，主收益标签为：

```text
r(e,x) = [O_x + sum(D_j, e < j <= x)] / O_e − 1
overnight_t = log[(O_t + D_t) / C_(t-1)]
```

进入除息日开盘的买方不获得当天分红；除息日开盘卖出的前日持有者获得。因此区间左开右闭。多日标签累加现金、不假设再投资；隔夜分量同步去掉机械除息跳空，日内GK项不变。策略逐日收益也包括应得分红，空头相应支付；诊断策略每天缩放名义敞口并复合收益，和固定股数持有多日标签不是同一投资规则。

**RET/RETX是收盘后资料。** 推导的分红金额不是在预测时已知的公告信息：`entry_*`分红字段只用于结果标签和事后诊断，模型特征中不使用。H2标签最早在实际退出开盘以及所需分红资料都已知后可入训练，即保守延迟至退出日收盘＋5分钟。预测CSV分别记录`target_exit_time`（真实持有期终点）和`target_end`（标签信息可得时点）；事件分组仍按前者，避免误把退出后的公告计入持有窗口。

总收益口径按除息日确认分红应收，没有模拟实际付款延迟、税收、资金成本或借券费用。旧版输入缺少整组分红列时仍运行价格收益，不会静默声称已调整；重建输入后metadata为`return_basis: cash_distribution_total_return`。

## 产物和后续运行

- `daily.csv`：直接给现有 `run`。
- `calendar.csv`：真实交易日表，含首个信号日前一日和末次入场日。
- `metadata.json`：来源、可得/成交时间假设、跨标的映射、局限。
- `market.csv`：尚未拼曲面的市场输入。
- `raw/`：保存三份原始输入快照。
- `build_manifest.json`：代码/原始文件哈希、节点缺失数、逐年曲面/偏斜差分覆盖率、现金分配推导检查及日历版本。

`quality.atm_call_put_diagnostic.maturities["30"]`（及`"91"`）包含50Δ call/put的配对天数、覆盖率、IV绝对差的均值/最大值。单位是年化小数IV；乘100为波动率点。缺50Δ put时统计为null且配对数为0，不当作差值为0，不改用其他delta。两侧50Δ未必是同一执行价，因此不是平价套利检验。

`quality.surface_coverage`和`skew_change_coverage_excluding_first`分开报告。缺一日节点可能损失两个偏斜差分，不插值补齐；研究报告另外检查整个开发期、包含预测开始之前的原始覆盖率，低于95%警告。不要只检查已有预测的覆盖率。

构建后抽样核对至少一个普通交易日、一个半日、夏令时前后，以及 SPX/SPY 身份映射。先冻结 `EXPERIMENT.md`，然后对足够长的历史运行：

```bash
python -m ivsurf run \
  --input data/real/build_01/daily.csv \
  --calendar data/real/build_01/calendar.csv \
  --metadata data/real/build_01/metadata.json \
  --phase development --out reports/wrds_development_01
```

默认需要多年数据，开发期包含两层顺序训练，最后252行保留。20天样本只执行 `build-wrds` 和 `validate`；不要为跑通而把训练窗缩成几天，再解释统计结果。

H2 的 h=1 是 t+1 开盘到 t+2 开盘。最后一行无未来退出价，研究引擎保留预测并将未知标签置空。软件未替用户获取真实WRDS数据；线上认证、订阅、实际表版本、数据发布延迟仍需在账户实测。

接口依据：[WRDS Python 客户端](https://github.com/wharton/wrds/blob/main/wrds/sql.py)、[exchange_calendars 日历实现](https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar.py)、[WRDS CRSP 字段说明示例](https://wrds-www.wharton.upenn.edu/demo/crsp/form/)。本地测试记录见 `VALIDATION.md`。
