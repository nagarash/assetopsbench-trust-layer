---
language:
  - en
pretty_name: AssetOpsBench
configs:
- config_name: scenarios
  data_files:
  - split: train
    path: data/scenarios/all_utterance.jsonl
  default: true
- config_name: compressor
  data_files:
  - split: train
    path: data/asset/compressor_utterance.jsonl
- config_name: hydrolic_pump
  data_files:
  - split: train
    path: data/asset/hydrolicpump_utterance.jsonl
- config_name: rule_logic
  data_files:
  - split: train
    path: data/task/rule_monitoring_scenarios.jsonl
- config_name: failure_mode_sensor_mapping
  data_files:
  - split: train
    path: data/task/failure_mapping_senarios.jsonl
- config_name: prognostics_and_health_management
  data_files:
  - split: train
    path: data/task/phm_utterance.jsonl
license: apache-2.0
task_categories:
- question-answering
- time-series-forecasting
tags:
- Industry
- PHM
- Predictive-Maintenance
- Asset-Management
- tool-learning
- task-automation
- LLM
- Multi-Agent
size_categories:
- n<1K
---

# AssetOpsBench

**AssetOpsBench** is a specialized benchmark designed for evaluating Large Language Models (LLMs) and Multi-Agent systems in industrial operations. It focuses on the intersection of sensor data interpretation, maintenance logic, and **Prognostics and Health Management (PHM)**.

The benchmark enables researchers to test how effectively AI agents can manage complex industrial assets, such as compressors and hydraulic pumps, by applying rule-based logic and diagnostic reasoning.

## 📂 Dataset Structure

The dataset is divided into several configurations to allow for granular testing. Users can load data for a specific **Asset** type or **Task** type.

### Baseline Configurations (Data Center Infrastructure)
This core set focuses on critical cooling systems within data center environments:
* **Asset Coverage**: Includes data from 4 Chillers and 2 Air-Handling Units (AHUs).
* **Lifecycle Tasks**: Benchmarks a model's ability to perform Anomaly Detection, Automated Sensor Mapping, and Work Order Generation.

### Asset Configurations
Focus on hardware-specific sensor patterns and operational contexts:
* **Compressor:** Data related to industrial air and gas compressors.
* **Hydrolic Pump:** Data focusing on fluid power systems and pressure diagnostics.

### Task Configurations
Focus on the reasoning and automation capabilities:
* **PHM (Prognostics and Health Management):** Tasks centered on predicting Remaining Useful Life (RUL) and assessing State of Health (SoH).
* **Rule Logic:** Evaluating the model's ability to trigger actions based on predefined industrial maintenance thresholds and logic.

## 🚀 Getting Started

You can load the default scenario or a specific configuration using the Hugging Face `datasets` library.

### Loading the Default Scenarios

```python
from datasets import load_dataset

dataset = load_dataset("ibm-research/AssetOpsBench", "scenarios")
```

### Loading a Specific Asset (e.g., Compressor)
```python
from datasets import load_dataset

dataset = load_dataset("ibm-research/AssetOpsBench", "compressor")
```

## Cite this Dataset
If you use our dataset in your paper, please cite our dataset by
```
@misc{patel2025assetopsbenchbenchmarkingaiagents,
      title={AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance}, 
      author={Dhaval Patel and Shuxin Lin and James Rayfield and Nianjun Zhou and Roman Vaculin and Natalia Martinez and Fearghal O'donncha and Jayant Kalagnanam},
      year={2025},
      eprint={2506.03828},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.03828}, 
}
```
