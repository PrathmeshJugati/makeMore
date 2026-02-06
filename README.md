This repository contains my personal implementation of the concepts from Andrej Karpathy's "Neural Networks: Zero to Hero" series. It focuses on building character-level language models from scratch, starting with simple statistics and evolving into modern deep learning architectures like WaveNet.

## 📂 Repository Structure

| File | Description | Concepts Covered |
| --- | --- | --- |
| **`bigram.ipynb`** | A simple Bigram Language Model. | Count-based probability, one-hot encoding, negative log-likelihood loss, basic PyTorch tensor manipulation. |
| **`mlp.ipynb`** | A Multi-Layer Perceptron (MLP) language model. | Implementation of **Bengio et al. 2003**, embeddings, hidden layers, non-linearities (`tanh`), and cross-entropy loss. |
| **`mlp_optim.ipynb`** | Optimization and initialization techniques. | Batch Normalization, Kaiming init, diagnostic tools for dead neurons, and preventing vanishing/exploding gradients. |
| **`mlp_wn.ipynb`** | A WaveNet-style architecture. | **Dilated Convolutions**, residual connections (skip connections), and hierarchical context fusion for processing longer sequences efficiently. |
| **`names.txt`** | The dataset. | A list of 32,000+ names used for training the models. |

## 🚀 Getting Started

### Prerequisites

To run these notebooks, you will need:

* Python 3.x
* PyTorch
* Matplotlib
* Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PrathmeshJugati/makeMore.git
cd makeMore

```


2. Install dependencies:
```bash
pip install torch matplotlib jupyter

```


3. Launch Jupyter Notebook:
```bash
jupyter notebook

```



## 🧠 Key Learnings

* **Bigram Model:** Understanding the baseline performance (loss ~2.45) and how simple statistical relationships drive prediction.
* **MLP (Bengio 2003):** Moving from integer counts to distributed representations (embeddings), allowing the model to generalize to unseen name combinations.
* **Internals:** How to visualize histograms of activations and gradients to debug "dead" neurons and saturated `tanh` layers.
* **WaveNet:** Refactoring flat list-based code into clean `torch.nn` modules and using dilated convolutions to "see" a larger context window without explosion in parameters.

## 🔗 References

* **Original Course:** [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
* **Dataset Source:** [ssa.gov](https://www.ssa.gov/oact/babynames/) (via Karpathy's repository)

---

*This project is for educational purposes, documenting my journey through deep learning fundamentals.*
