# Pinn-Torch

This repository contains resources related to the paper:

**#Neural Network Models as Surrogates of Classical Ordinary Differential Equations for Reduced-Order Single-Cell Electrophysiology**
>Authors: Yan B. Werneck, Bernardo M. Rocha, Rafael Sachetto Oliveira, Rodrigo W. dos Santos
Affiliations: Federal University of Juiz de Fora (UFJF), Federal University of São João del-Rei (UFSJ).
(source : https://repositorio.ufjf.br/jspui/handle/ufjf/18126)
---

## 🚀 Workflow

The system is **modular**: you combine
- a **Model** (`Net.py`)
- one or more **Loss Functions** (`Loss.py` or `Loss_PINN.py`)
- inside a **Trainer** (`Trainer.py`)

The **Trainer** orchestrates the entire process of training and output generation.


    A[Training Data (.npy/.csv)] --> C{Trainer};
    B[Network Architecture (Net)] --> C;
    subgraph "Loss Functions"
        D1[Data-Driven Loss (MSE, MAE)]
        D2[Physics-Informed Loss (PINN)]
    end
    D1 --> C;
    D2 --> C;
    C -- Starts Training --> E((Execution));
    E -- Produces Outputs --> F[Loss Logs (losses.csv)];
    E -- Produces Outputs --> G[Best Model (best_model.pth)];
    E -- Produces Outputs --> H[Gradient Logs (model_stats.txt)];


## 🧩 Step 1 — Define the Architecture (Net.py) ##

The FullyConnectedNetworkMod class allows customization of each network layer: number of neurons + activation function.


| Input         | Output        |
| ------------- |:-------------:|
| list of tuples `[(Activation, N_neurons), ...]`  | PyTorch model `(nn.Module)` ready for training.     |


```
#main_script.py
from fisiocomPinn.Net import FullyConnectedNetworkMod

# Define architecture: 3 hidden layer


hidden_config = [("ReLU", 128), ("ReLU", 128), ("Tanh", 64)]

# Problem C from the paper:
input_dim = 4   # Input: [t, u0, w0, Iiapp]
output_dim = 2  # Output: [u(t), w(t)]

my_model = FullyConnectedNetworkMod(
    input_shape=input_dim,
    output_shape=output_dim,
    hidden_config=hidden_config
)

print(my_model)

```


# 🎯 Step 2 — Configure the Trainer (Trainer.py) #

The Trainer is the central class. It takes the model, losses, and manages the training process.

## 🔹 Scenario A: Data-Driven (DDNN / ITNN) ##

The model learns input → output mappings from numerical solver data.

```python
from fisiocomPinn.Trainer import Trainer
from fisiocomPinn.Loss import MSE
import torch

\# Load data
input_data = torch.load("inputs.pt")  # shape [N, 4]
target_data = torch.load("targets.pt") # shape [N, 2]

\# 1. Instantiate Trainer
trainer = Trainer(model=my_model, output_folder="results_ddnn")

\# 2. Define loss (MSE)
loss_mse = MSE(data_in=input_data, target=target_data, batch_size=1024)

\# 3. Add loss to Trainer
trainer.add_loss(loss_mse, weight=1.0)

\# 4. Start training
\# trainer.train(num_iterations=200000)
```

## 🔹 Scenario B: Physics-Informed (PINN) ##

Here, the loss is computed from the ODE residuals (via autograd).

```from fisiocomPinn.Trainer import Trainer
from fisiocomPinn.Loss_PINN import LOSS_PINN
import torch

\# --- Batch generator ---
def batch_generator(batch_size, device):
    t = torch.rand(batch_size, 1, device=device) * 50.0
    t.requires_grad = True
    return t

\# --- FitzHugh-Nagumo residual function ---
def fhn_pinn_func(t_batch, model):
    a, b, c, tau, Iiapp = 0.7, 0.8, 1.0, 12.5, 0.5
    uv_pred = model(t_batch)
    u, w = uv_pred[:, 0:1], uv_pred[:, 1:2]

    du_dt = torch.autograd.grad(u, t_batch, torch.ones_like(u), create_graph=True)[0]
    dw_dt = torch.autograd.grad(w, t_batch, torch.ones_like(w), create_graph=True)[0]

    residual_u = du_dt - (c * (u - u**3 / 3 - w) + Iiapp)
    residual_w = dw_dt - ((u + a - b * w) / tau)

    return torch.mean(residual_u**2) + torch.mean(residual_w**2)

\# --- Trainer setup ---
trainer_pinn = Trainer(model=my_model, output_folder="results_pinn")

loss_pinn = LOSS_PINN(
    pinn_func=fhn_pinn_func,
    batch_generator=batch_generator,
    batch_size=2048,
    name="FHN_Residual_Loss"
)

trainer_pinn.add_loss(loss_pinn, weight=1.0)

\# trainer_pinn.train(num_iterations=200000)
```

# 📊 Step 3 — Train and Analyze #

During training, the following outputs are generated inside output_folder/:

|Output       |Explanation       |
| ------------- |:-------------:|
| best_model.pth   →    | Model weights with lowest validation error      |
| losses.csv → | CSV log of loss history (useful for convergence plots).|
|model_stats.txt  → | Stats about weights/gradients (useful to diagnose vanishing/exploding gradients).|


## Inference with Trained Model ##
```python
from fisiocomPinn.Net import FullyConnectedNetworkMod
import torch

\# Recreate the same architecture
my_model = ...

\# Load trained weights
my_model.load_state_dict(torch.load("results_ddnn/best_model.pth"))
my_model.eval()

\# Prediction
\# input_tensor = ...
\# predicted_output = my_model(input_tensor)
```


<!-- ## 📖 Abstract

Electrophysiology modeling plays a crucial role in non-invasive diagnostics and in advancing our understanding of cardiac and brain function. Traditional methods rely on solving systems of **ordinary differential equations (ODEs)**, which are computationally expensive.

This study explores **neural networks as differentiable surrogate models** for electrophysiology simulations, using the **FitzHugh–Nagumo (FHN)** model as a case study. Three surrogate strategies are investigated:

- **Data-Driven Neural Networks (DDNNs):** trained directly on numerical solutions.
- **Physics-Informed Neural Networks (PINNs):** integrate ODE constraints into the loss function.
- **Iterative Neural Networks (ITNNs):** learn discrete update rules for advancing system states.

With **TensorRT optimization**, surrogates achieve up to **1.8× speedup** compared to optimized CUDA solvers, with minimal accuracy loss.

---

## ⚙️ Methodology

- **Base model:** FitzHugh–Nagumo (FHN), describing excitable cell dynamics.
- **Problem settings:**
  - **A:** solution depends only on time `t`.
  - **B:** adds initial conditions `(u0, w0)`.
  - **C:** adds external current `Iiapp`.
- **Data generation:** Euler method with 0.01 ms time step.
- **Training sets:** 1k, 10k, and 100k samples.
- **Hardware:** NVIDIA RTX 4070 GPU + Intel i5-12400F CPU.
- **Optimization:** TensorRT for GPU-accelerated inference.

---

## 📊 Results

- **DDNNs** → best accuracy in data-rich scenarios, robust and faster to train.
- **PINNs** → effective only in data-scarce scenarios, but ~2× higher training cost.
- **ITNNs** → prone to instability due to error accumulation across iterations.

➡️ **Conclusion:**
- DDNNs are the most practical surrogates when sufficient data can be generated.
- PINNs help when data is scarce, but add computational overhead.
- ITNNs face limitations in long-term stability.

---

## 🚀 Applications

- **Digital twins** for personalized cardiology and neurology.
- **Real-time simulations** for clinical decision support.
- **Differentiable AI pipelines** for scientific computing.

--- -->


## Citation

If you use this work, please cite it as:

@article{werneck2025neural,
author = {Yan Barbosa Werneck and others},
title = {Neural network surrogates for the FitzHugh–Nagumo model},
<!-- year = {2025}, -->
<!-- journal = {To be updated}, -->
note = {Preprint},
<!-- url = {https://arxiv.org/abs/xxxx.xxxxx} -->


---

## Acknowledgments

The authors would like to thank the following institutions for their support:

- **FAPEMIG** (Minas Gerais State Research Support Foundation)
- **CAPES** (Coordenação de Aperfeiçoamento de Emprego de Nível Superior)
- **CNPq** (National Council for Scientific and Technological Development) — Proc. 423278/2021-5 and 310722/2021-7
- **Ebserh** (Empresa Brasileira de Serviços Hospitalares)
- **SINAPAD Santos-Dumont**
- **Federal University of Juiz de Fora**

---

## Contact

For correspondence, please reach out to:

- **Name:** Yan Barbosa Werneck
<!-- - **Address:** Rua São Mateus 872, Juiz de Fora, Brazil   -->
- **Email:** [yanbwerneck@outlook.com](mailto:yanbwerneck@outlook.com)
