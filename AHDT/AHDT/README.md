# AHDT

Adaptive Hybrid-feature Decision Tree（AHDT）

## 目录结构

```text
AHDT/
├── model.py                 # AHDT 模型主体：DR、ReliefF、WDR、树构建与预测
├── data.py                  # 通用表格分类数据读取框架
├── evaluation.py            # 分层 K 折交叉验证和分类报告
├── run_experiment.py        # 通用命令行实验入口
├── examples/
│   └── bank_marketing.py    # Bank Marketing 数据集示例
├── tests/
│   └── test_smoke.py        # 基本训练/预测测试
├── requirements.txt
└── README.md
```

## 安装

```bash
cd AHDT
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 运行通用实验

```bash
python run_experiment.py \\
  --data ./数据集/bank/bank.csv \\
  --target y \\
  --delimiter ';'
```

## 运行 Bank Marketing 示例

```bash
python examples/bank_marketing.py \\
  --data ./数据集/bank/bank.csv
```

如果使用 `bank-full.csv`，只需要替换 `--data` 路径。

## 运行测试

在 `AHDT` 目录中运行：

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 作为 Python 模块使用

```python
from model import AHDT
from data import load_tabular_classification_data
from evaluation import cross_validate_ahdt

X, y, feature_names, metadata = load_tabular_classification_data(
    "your_dataset.csv",
    target_column="label",
    delimiter=",",
)

model = AHDT(
    alpha_strategy="auto",
    k_range=(2, 6),
    quantile_threshold=0.6,
    max_depth=8,
    min_samples_split=20,
    random_state=42,
)
model.fit(X, y)
predictions = model.predict(X)
```

在其他文件夹中导入时，可以使用：

```python
from AHDT.model import AHDT
```

## 说明

`tests/test_smoke.py` 是轻量测试，不代表论文实验结果。

其余数据集的下载去UCI官网