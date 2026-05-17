# 👋 Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation

**Official Code for the paper accepted to *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)***

---

## 🧪 Getting Started

### ⚙️ Environment Setup

Ensure you have [Conda](https://docs.conda.io/) installed. Then, create and activate the environment using the provided file:

```bash
conda env create --name env.fl --file=environment.yml
conda activate env.fl
```

### 🚀 Running the Code

First, set up the HuggingFace Accelerate configuration:

```bash
cp accelerate_default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

Next, launch the training script for the CIFAR-100 dataset:

```bash
bash run-cifar100.sh
```

---

## 📁 Project Structure

```
.
├── algorithms/
│   ├── engine/   # Federated learning coordination logic
│   └── solver/   # Local training procedures
├── config/         # YAML configuration files
├── data/           # Dataset cache directory
├── log/            # Output logs and saved results
├── model/          # Model definitions
├── utils/          # Utility functions
├── main.py         # Entry point for training
└── test.py         # Evaluation and testing routines
```

---

## 📄 Citation

If you find this work useful for your research, please cite our paper:

```bibtext
@article{zhang2025fed,
  title={Fed-hello: Efficient federated foundation model fine-tuning with heterogeneous lora allocation},
  author={Zhang, Zikai and Liu, Ping and Xu, Jiahao and Hu, Rui},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  publisher={IEEE}
}
```

---

## 🔗 Our Federated Fine-Tuning Series

This repository is part of our broader research effort on efficient and heterogeneous federated fine-tuning of foundation models. Our recent works explore federated fine-tuning from multiple complementary perspectives, including rank-wise adaptation, layer-wise LoRA allocation, and cross-domain benchmarking.

### Rank-wise Federated Fine-Tuning

- **Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation**  
  *ICLR 2026*  
  We propose a rank-wise federated fine-tuning method that mitigates initialization noise and aggregation noise under heterogeneous client settings.  
  [[Paper](https://arxiv.org/pdf/2602.16936)] [[Code](https://github.com/TNI-playground/Fed-PLoRA)]

### Layer-wise Federated Fine-Tuning

- **Fed-HeLLo: Efficient Federated Foundation Model Fine-Tuning with Heterogeneous LoRA Allocation (this repo)**  
  *IEEE TNNLS 2025*  
  We study heterogeneous layer-wise LoRA allocation for efficient federated fine-tuning of foundation models.  
  [[Paper](https://arxiv.org/pdf/2506.12213)]

- **Fed-pilot: Optimizing LoRA Allocation for Efficient Federated Fine-Tuning with Heterogeneous Clients**  
  *ArXiv 2024*  
  We introduce an optimization framework for allocating LoRA modules across layers under heterogeneous client resources.  
  [[Paper](https://arxiv.org/pdf/2410.10200)]

### Benchmark

- **FlowerTune: A Cross-Domain Benchmark for Federated Fine-Tuning of Large Language Models**  
  *NeurIPS 2025, Datasets and Benchmarks Track*  
  In collaboration with FlowerLabs, we build a cross-domain benchmark for evaluating federated fine-tuning of large language models.  
  [[Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/563991b5c8b45fe75bea42db738223b2-Paper-Datasets_and_Benchmarks_Track.pdf)] [[Project](https://flower.ai/docs/examples/flowertune-llm.html)]

---

## 📬 Contact

For any questions or suggestions, please feel free to open an issue on this repository or contact the authors directly.
