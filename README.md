# One-Solves-All

**A Prompt Framework for Expert-Free LLM-Based Fully Automated Simple Power Analysis on Cryptosystems**

> Wenquan Zhou, An Wang, Yaoling Ding*, Congming Wei, Jingqi Zhang, Jiakun Li, Liehuang Zhu
>
> *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)*

## Overview

Side-channel analysis (SCA) is a powerful technique to extract secret data from cryptographic devices, but it heavily relies on domain experts and specialized tools. This work explores the potential of Large Language Models (LLMs) to bridge this gap, enabling **non-experts** to perform fully automated Simple Power Analysis (SPA) on public-key cryptosystems through **a single prompt interaction**.

We propose a novel **prompt framework** specifically designed for SPA tasks, consisting of six components:

- **Role**: Assigns the LLM an expert persona in side-channel analysis
- **Reinforce**: Strengthens autonomous behavior without user intervention
- **Input**: Specifies trace file path and parameters
- **Task Description**: Defines the classification and output requirements
- **Chain-of-Thought (CoT)**: Guides step-by-step reasoning (segmentation, dimensionality reduction, clustering)
- **Expert Strategies**: Three domain-specific strategies for robust SPA

## Key Results

| Model | Overall Accuracy | vs. Baseline |
|-------|:---:|:---:|
| GPT-4o | **98.02%** | +45.98% |
| DeepSeek-V3.1 | **97.83%** | +45.79% |
| Unsupervised HCA (Baseline) | 52.04% | - |

- Evaluated on **25 power traces**: 7 real traces + 18 simulated traces
- Covers **RSA**, **ECC**, and **Kyber** (ML-KEM) implementations
- Reduces analysis time by **over 93.09%** compared to manual analysis
- First successful application of LLMs to achieve fully automated SPA

## Architecture

The system uses an **LLM-based agent** with a Jupyter kernel backend:

- `langchain_tool_calling__agent_with_md.py` — Main agent script implementing the prompt framework with LangChain. The LLM performs SPA by executing Python code through a Jupyter API tool.
- `python_jupyter_kernel_tool.py` — Flask-based Jupyter kernel service that executes Python code and returns results to the agent.
- `trace_data/` — Power trace dataset (`.npy` files) from real cryptographic devices and simulated leakage patterns.

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-openai flask jupyter_client pydantic requests matplotlib numpy scikit-learn
```

### 2. Configure LLM API

Edit `langchain_tool_calling__agent_with_md.py` and set your API key and endpoint:

```python
siliconflow_api = "https://api.siliconflow.cn/v1"
siliconflow_key = "your-api-key-here"
```

Or use OpenRouter:

```python
openrouter_api = "https://openrouter.ai/api/v1"
openrouter_key = "your-api-key-here"
```

### 3. Run

```bash
python langchain_tool_calling__agent_with_md.py
```

The agent will automatically:
1. Start the Jupyter kernel backend
2. Load the power trace file
3. Perform segmentation, dimensionality reduction, and clustering
4. Verify constraints and iterate if needed
5. Output the recovered operation sequence

## Dataset

The `trace_data/` directory contains power traces from:

| Target | Platform | File |
|--------|----------|------|
| ECC | AT89S52 | `ECC-AT89S52-middle-6945点.npy` |
| ECC | Smart Card | `ECC-smart_card-10000点.npy` |
| RSA | ASIC | `RSA-ASIC Y-middle-5210点.npy` |
| RSA | SAKURA-G | `RSA-SAKURA-G-middle-57285点.npy` |
| RSA | STM32F429 | `RSA-STM32F429-9030点.npy` |
| RSA | Smart Card | `RSA-smart_card-middle-11702点.npy` |
| Kyber | DFt2 | `kyber-DFt2-middle-4727点.npy` |

Plus 18 simulated traces covering typical SPA leakage patterns.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2025one,
  title={A Prompt Framework for Expert-Free LLM-Based Fully Automated Simple Power Analysis on Cryptosystems},
  author={Zhou, Wenquan and Wang, An and Ding, Yaoling and Wei, Congming and Zhang, Jingqi and Li, Jiakun and Zhu, Liehuang},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2025}
}
```

## License

This project is for academic research purposes.
