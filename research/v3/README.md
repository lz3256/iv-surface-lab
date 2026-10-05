# IV Surface v3 探索包

先看 **研究说明.md**。结果全部为探索性；未购买新样本。

主要文件：

- PROTOCOL.md / protocol_freeze.json：本次预测前固定的规则与本地文件哈希。
- results/comparisons.csv：全部 R3u 与同端点配对比较，增量 R² 用小数比例存储，乘 100 后才是百分数。
- results/v1_v2_saved_interval_conversion.csv：旧置信区间按固定样本基线 MSE 换算。
- results/v1_v2_ratio_bootstrap.csv：旧预测重新联合抽样损失差与分母。通用计算函数保留阈值字段；本研究只在 horizon=1 解释 0.1% 阈值，其他期限不作阈值判断。
- results/former_*/：旧 R0–R3、新增预测及残差化参数、训练行审计。
- diagnostics/：同端点差分、失效原因、配对日历分组和切换前后统计。
- inputs/events.csv：官方事件日期、时间、来源及未核验时刻标记。
- events/：事件窗口映射、逐次预测、训练审计和两项比较。
- code/：原 v3.2 引擎及新增脚本；audit_extension.py、switch_alignment.py 是结果后的核验/描述脚本，不属于预测前冻结的模型代码。
- v1_snapshot/、v2_snapshot/：独立复现所需旧输入与旧预测。不是全量原始期权报价档案。
- tests.log / verification.json / PACKAGE_SHA256.json：测试、审计及交付文件哈希。

Python 3.10+，依赖为 NumPy、pandas。建议独立虚拟环境。以下命令从包根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
cd code
python -m unittest test_extension -v
python audit_extension.py --v2 ../v2_snapshot --package ..
```

完整复现：先复制整个交付目录作为实验副本，将副本内原 results/ 与 events/ 重命名留存（run_daily.py 拒绝写入非空结果目录）。然后在副本 code/ 下执行：

```bash
python same_endpoints.py --v1 ../v1_snapshot --out ../diagnostics
python run_daily.py --v1 ../v1_snapshot --v2 ../v2_snapshot --package ..
python run_events.py --v2 ../v2_snapshot --package ..
python switch_alignment.py --package ..
python audit_extension.py --v2 ../v2_snapshot --package ..
```

计算不需要联网。事件 CSV 是固定输入；官方日期构建依据可按每行链接回查。日历已修订版本是否当时可得未被独立确认，不能作为已验证的实时交易日历。不同数值库版本可能有末位浮点差异。

protocol_freeze.json 的旧文件路径是原工作目录的相对路径；便携包审计会跳过不存在的外部旧文件，随包文件由 PACKAGE_SHA256.json 覆盖。不要把本地冻结哈希等同于外部预注册。
