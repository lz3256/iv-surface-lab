# IV Surface：现有数据运行包

先阅读 `研究结果.md`，或打开 `index.html`。开发期与留出期的完整 HTML 报告位于 `reports/development/report.html` 和 `reports/final/report.html`。

## 最新补充

`补充解读.md` 包含IV水平的逐年稳定性和收益模型相对简单均值基线的比较；对应数值见 `iv_level_yearly.csv`。`数据核验.md` 汇总原始输入的问题。

本包包含可复现模型结果的处理后输入、代码、报告和审计记录。原始optionsDX大压缩包保留在你的数据目录；本包记录其哈希及来源文件名。

## 本次处理

使用用户提供的 optionsDX SPX 日频期权链和 CRSP SPY 数据。原始下载文件未修改。完整交易日历保留，非交易日快照隔离，缺失曲面不跨日补值。

预先进行的数据质量检查发现，2019年5月6个日期的50Δ call/put IV差异超过20个波动率百分点。对应日期的四个主节点设为缺失，原始构建值保存在 `data/surface.csv`，排除记录见 `data/curve_exclusions.csv`。这一规则在预测结果运行前记录，见 `PROTOCOL.md` 的修订说明和 `protocol_freeze_amended.json`。

合约结算类型、供应商DTE和历史数据可得时点未被独立核实。因此结果属于探索性研究，不是OptionMetrics复现或实际可执行交易业绩。

## 文件

- `研究结果.md`：中文核心结果与限制。
- `results_summary.csv`：两阶段关键模型比较。
- `verification.json`：训练时间和独立收益标签核验。
- `PROTOCOL.md`：数据口径、时间假设、模型参数及评估划分。
- `data/daily.csv`：模型输入，含完整交易日历上的缺失曲面行。
- `data/surface.csv`：清洗极端call/put差异前的供应商期限曲面节点。
- `data/expiry_nodes.csv`：各日期各到期合约的Delta节点。
- `data/interpolation_audit.csv`：固定期限插值所用的上下界。
- `data/build_manifest.json`：原始文件哈希、构建记录、排除日期和假设。
- `reports/development`、`reports/final`：全部模型指标、逐年比较、逐日预测、HAR训练审计和假设策略损益。
- `code/build_optionsdx.py`：原始压缩包到模型输入的适配器。
- `code/ivsurf`：本次实际使用的研究引擎副本。
- `tests.log`：64项测试的运行记录。

## 复现预测（使用已构建的数据）

在此目录运行，使用已安装 NumPy 和 pandas 的 Python：

```bash
PYTHONPATH=code python3 -m ivsurf run \
  --input data/daily.csv --calendar data/calendar.csv \
  --metadata data/metadata.json --phase development \
  --out reports/reproduction_development

PYTHONPATH=code python3 -m ivsurf run \
  --input data/daily.csv --calendar data/calendar.csv \
  --metadata data/metadata.json --phase final \
  --out reports/reproduction_final
```

输出目录必须是新的，避免覆盖已有实验。默认参数即本次协议参数。此处的 final 是固定时间留出评估，允许按时间顺序使用已经知道的早期留出标签重新训练，不能使用未来标签。

从压缩包重建输入另需 `exchange_calendars` 和支持7z解压的 `bsdtar`，运行 `code/build_optionsdx.py --help` 查看入口。该适配器的CRSP源文件名对应本次导出的 `if92ihpkzaflamwz.csv`。

## 解释

MSE改善为正表示新增特征改善预测；为负表示比基线更差。收益预测与波动率预测是两个不同问题。所有结果都应结合原始p值、多重比较调整、跨年稳定性和数据限制阅读。

留出期已在本次评估中使用。未来根据这些结果改模型，需要新的未见数据才能再次做独立确认。重复运行相同输入仅是技术复现，不能增加统计证据。
