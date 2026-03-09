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

## 📬 Contact

For any questions or suggestions, please feel free to open an issue on this repository or contact the authors directly.
