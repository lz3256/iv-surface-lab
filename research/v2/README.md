# IV Surface v2 探索性补丁包

先读 **研究说明.md**。这是 H2 稳健性补丁、结算周期诊断及一次固定规则重跑；不是新确认研究，也没有识别或修复 AM/PM。

- code/：原 v3.2 引擎及新增脚本；H1 引擎未改。
- PROTOCOL.md、protocol_freeze.json：此次模型运行前的探索性规则和本地哈希。
- results/：新预测、训练审计、误差与全部比较。
- diagnostics/：旧模型危机期损失分解、训练窗口 z 分数、第三周五分组及到期组合诊断。
- verification.json、tests.log：核验结果。
- v1_snapshot/：复现所需的旧输入与旧预测快照；不含全量原始期权档案，也不能据此从原始报价重建曲面。
- h2_exploratory.patch：向旧项目添加脚本与独立协议，不覆盖旧引擎。
- PACKAGE_SHA256.json：包内文件清单与哈希。

从本目录运行，需要 Python 3.10+；依赖见 code/requirements.txt。建议在独立虚拟环境安装。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
cd code
python -m unittest test_robust_h2 -v
python diagnose_v1.py --v1 ../v1_snapshot --out ../diagnostics_reproduced
python run_v2.py --v1 ../v1_snapshot --out ../results_reproduced
python audit_v2.py --v1 ../v1_snapshot --package ..
```

run_v2.py 要求输出目录为空或不存在。audit_v2.py 审计随包 results/；若要审计重跑目录，可先在包副本里把新结果目录换成 results/。不同 NumPy/BLAS 版本可能产生末位数值差别。

也可对完整旧包应用补丁：在旧包根目录执行 `git apply --check /path/to/h2_exploratory.patch`，再 `git apply /path/to/h2_exploratory.patch`。随后进入 code/，用 `python run_v2.py --v1 .. --out ../v2_results` 运行；诊断用 `python diagnose_v1.py --v1 .. --out ../v2_diagnostics`。补丁只包含新分析模块、测试和 V2_EXPLORATORY_PROTOCOL.md；独立交付包的核验脚本与快照请在本包中使用。

原始 SPX/SPXW 根代码、结算时刻及文件历史可用时间仍未核验；前日文件次日 08:00 可得和开盘成交仍是旧研究假设。价格和衍生数据随包仅供你的本地复现使用，向外分享前需按数据供应方授权处理。
