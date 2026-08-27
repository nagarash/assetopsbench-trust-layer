---
license: apache-2.0
tags:
  - industrial-maintenance
  - question-answering
  - benchmark
language:
  - en
---

# Dataset Card for ibm-research/AssetOpsBench

AssetOpsBench is a curated benchmark dataset for evaluating agentic systems and large language models (LLMs) in **industrial asset operations**, particularly focusing on tasks like condition monitoring, failure analysis, and sensor-based diagnostics.

The dataset provides structured question-answering and multi-agent scenario prompts based on real-world operational contexts, such as HVAC systems (e.g., chillers, air handling units), and is part of the AssetOpsBench framework developed at IBM Research.

---

## Dataset Summary

This dataset contains two major categories:

1. **Scenario QA (`scenarios`)**:
   Human-authored evaluation prompts designed to test the performance of multi-agent orchestration strategies.

2. **FailureSensorIQ (`failuresensoriq_...`)**:
   A family of question-answering datasets focused on sensor-failure relationships, supporting multiple variants for robustness and evaluation.

---

## Supported Configurations

| Config Name                        | Type             | Description                                                    |
|-----------------------------------|------------------|----------------------------------------------------------------|
| `scenarios`                       | Scenario QA      | Human-authored prompt scenarios for agent orchestration        |
| `failuresensoriq_all`             | QA Dataset       | Full dataset with diverse QA covering failure-sensor mappings  |
| `failuresensoriq_10_options`      | QA Dataset       | 10-option MCQ format                                           |
| `failuresensoriq_perturbed_complex` | QA Dataset     | Complex perturbed choices for adversarial robustness           |
| `failuresensoriq_perturbed_simple` | QA Dataset      | Simpler perturbed options                                      |
| `failuresensoriq_multi_answers`   | QA Dataset       | Questions with multiple correct options                        |
| `failuresensoriq_sample_50`       | QA Subset        | ⭐ Expert-authored subset of 50 questions for benchmarking      |

---

## Languages

- English

---

## Use Cases

- Benchmarking LLMs (GPT, LLaMA, Claude, etc.)
- Evaluating multi-agent reasoning systems
- Industrial diagnostic model prototyping
- Agent planning and verification
- Robustness testing via perturbed QA

---

## Dataset Structure

Each dataset split is provided in `.jsonl` format where each line is a QA pair or prompt dictionary. Common fields include:

- `question`: The user query
- `options`: List of possible answers (if applicable)
- `answer`: Correct answer(s)
- `metadata`: (Optional) Includes asset ID, failure mode, or sensor references

---

## Citation

If you use this dataset, please cite:

@misc{patel2025assetopsbenchbenchmarkingaiagents,
      title={AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance}, 
      author={Dhaval Patel and Shuxin Lin and James Rayfield and Nianjun Zhou and Roman Vaculin and Natalia Martinez and Fearghal O'donncha and Jayant Kalagnanam},
      year={2025},
      eprint={2506.03828},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.03828}, 
}


---

## License

Apache 2.0

---

## Source and Acknowledgements

Developed by [IBM Research](https://research.ibm.com/) as part of the [AssetOpsBench](https://github.com/ibm-research/AssetOpsBench) project.

Special thanks to domain experts who authored the high-quality benchmark subset (`failuresensoriq_sample_50`).

---

## Tags

- domain: industrial-maintenance
- type: expert-curated, QA, benchmark
- task: question-answering, multi-agent evaluation
- modality: text, tabular, sensor
