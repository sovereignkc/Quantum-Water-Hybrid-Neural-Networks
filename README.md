# Quantum-Water-Hybrid-Neural-Networks
A Quantum Mathematical Neural Network Insight into The Advantages of Quantum Hybrid Neural Networks because they leverage quantum states to compactly represent complex fluid behaviors and train with significantly fewer parameters. The models here are trained for leak and burst deteciton, and pressure (PSIG stands for Pounds per Square Inch Gauge)

# First Dataset (0.1 Loss on Test)
- [Pre Desalienation RO Feedwater Performance Data](https://www.kaggle.com/datasets/saurabhshahane/ro-feedwater-performance-data)

# Second Dataset (Water Leak Dataset) (99% Leak Detection Accuracy, 99.5% Burst Detection Accuracy)
- [Water Leak Dataset](https://www.kaggle.com/datasets/ziya07/water-leak-dataset)


# 99% Leak / 99.5% Burst Detection Architecture Explanaiton and Walkthrough
# QuantumAttentionInfrastructureGuard: Mathematical & Architectural Analysis


## 1. High-Level System Workflow

The architecture routes input features through a sequential pipeline designed to isolate signal anomalies before processing them inside a dual-topology quantum Hilbert space.

```
Input Features (X) ──> HydroFeatureAttention ──> GatedResidualBlocks ──> Linear Projections
                                                                                 │
┌─────────────────────── Dual-Stream Variational Quantum Processing ─────────────┴───────────┐
│                                                                                           │
│  Stream 1:  ──> Tanh & Scale ([-π, π]) ──> Dropout ──> QNode 1 (VQC Base) ──> Expectation  │
│                                                                                 │         │
│  Stream 2:  ──> Tanh & Scale ([-π, π]) ──> Dropout ──> QNode 2 (VQC Base) ──> Expectation  │
│                                                                                 │         │
└─────────────────────────────────────────────────────────────────────────────────┼─────────┘
                                                                                  ▼
                                         Output Logits <── Decoder <── Concatenate Signals [q1_out, q2_out]

```

---

## 2. Mathematical Formalization

### 1. Tabular Self-Attention (`HydroFeatureAttention`)

Given an input vector $x \in \mathbb{R}^{D}$, the vector is unsqueezed into an explicit tabular sequence representation:


$$X \in \mathbb{R}^{D \times 1}$$

For each feature index $i$, linear projections map the scalar feature to a latent space of dimensionality $d_k$ (where `head_dim` $= 1$):


$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$


Where $W_Q, W_K, W_V \in \mathbb{R}^{1 \times d_k}$. The scaled dot-product attention matrix $A \in \mathbb{R}^{D \times D}$ calculates cross-feature correlations:


$$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$


The attention context matrix $H$ is computed and projected back to the original feature dimensions via $W_O \in \mathbb{R}^{d_k \times 1}$:


$$H = A V, \quad x_{\text{attn}} = \text{squeeze}(HW_O) \in \mathbb{R}^{D}$$

### 2. Gated Residual Block Processing (`GatedResidualBlock`)

The transformed vector passes through deep feed-forward structures equipped with a multiplicative Gated Linear Unit (GLU) variation:


$$\tilde{x} = W_2 \cdot \text{ReLU}(W_1 x_{\text{attn}} + b_1) + b_2$$

$$g = \sigma(W_g \tilde{x} + b_g)$$

$$x_{\text{res}} = (\tilde{x} \odot g) + \text{skip}(x_{\text{attn}})$$


Where $\odot$ represents the Hadamard (element-wise) product, $\sigma$ is the standard sigmoid activation function, and $\text{skip}(\cdot)$ is an identity mapping or linear projection to adjust dimensionality.

### 3. State Preparation & Variational Quantum Circuit (VQC)

The encoder outputs a hidden classical representation $h \in \mathbb{R}^{16}$. Two separate linear layers project this representation into two 4-dimensional spaces ($n=4$):


$$\phi_1 = \tanh(W_{q1} h + b_{q1}) \cdot \pi, \quad \phi_2 = \tanh(W_{q2} h + b_{q2}) \cdot \pi$$


Both vectors $\phi_1, \phi_2 \in [-\pi, \pi]^4$ parameterize the state preparation phase of the VQCs.

#### Quantum Information Flow (Per QNode)

1. **State Preparation (Angle Embedding):** The initial state $|0\rangle^{\otimes n}$ is rotated using $Y$-axis rotations:

$$|\psi_0(\phi)\rangle = \bigotimes_{i=0}^{n-1} R_Y(\phi_i)|0\rangle$$


2. **Variational Topology Block 1:** Applies a strongly entangling layer parameterized by weight tensor $\theta^{(0)}$:

$$|\psi_1\rangle = U_{\text{Entangling}}(\theta^{(0)}) |\psi_0(\phi)\rangle$$


3. **Internal Mid-Circuit Topology Routing:** Explicit linear CNOT chains form non-local entanglement structures across the register:

$$U_{\text{CNOT}} = \prod_{i=0}^{n-2} \text{CNOT}_{i, i+1}$$


$$|\psi_2\rangle = U_{\text{CNOT}} |\psi_1\rangle$$


4. **Variational Topology Block 2:** Applies a second strongly entangling layer parameterized by weight tensor $\theta^{(1)}$:

$$|\label{eq:final_state} \psi_{\text{final}}\rangle = U_{\text{Entangling}}(\theta^{(1)}) |\psi_2\rangle$$


5. **Expectation Value Measurement:**
The QNode returns an array of expectation values calculated against the Pauli-Z operator for every individual wire $i$:

$$\langle Z_i \rangle = \langle \psi_{\text{final}} | Z_i | \psi_{\text{final}} \rangle \in [-1, 1]$$



### 4. Classification Logits & Loss Header

The output streams from both quantum chips are concatenated to construct an enriched 8-dimensional signal vector:


$$h_{\text{quantum}} = [\langle Z_0 \rangle_1, \dots, \langle Z_3 \rangle_1, \langle Z_0 \rangle_2, \dots, \langle Z_3 \rangle_2]^T \in \mathbb{R}^8$$


This vector passes through the decoder block to generate a raw, unbounded classical logit $\hat{y} \in \mathbb{R}$.

The architecture eliminates the final sigmoid operation from the forward pass, using **BCEWithLogitsLoss** instead. This prevents gradient vanishing during extreme anomalies (such as a 99.5% burst scenario). The objective function is defined as:


$$\mathcal{L} = -\frac{1}{N}\sum_{j=1}^{N} \left[ y_j \cdot \log \sigma(\hat{y}_j) + (1 - y_j) \cdot \log (1 - \sigma(\hat{y}_j)) \right]$$

---

## 3. Explaining Important Code Blocks

### 1. The Variational Quantum Circuit Definition

```python
def vqc_base(inputs, weights):
    # Initial data angle mapping
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

    # First variational entangling layer block
    qml.templates.StronglyEntanglingLayers(weights[0], wires=range(n_qubits))

    # Internal mid-circuit entanglement routing
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    # Final variational processing layer block
    qml.templates.StronglyEntanglingLayers(weights[1], wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

```

* **Why it matters:** This function defines the layout of our quantum circuit. It maps classical features directly to rotation angles ($\theta = \phi_i$) on a Bloch sphere using $Y$-axis rotations.
* **Key Mechanism:** It uses a "sandwich" design. A manual CNOT entangling chain is placed between two separate `StronglyEntanglingLayers` parameter blocks. This structure creates complex quantum states across all four qubits, maximizing the expressiveness of the quantum state space.

---

### 2. Tabular Self-Attention Layer

```python
class HydroFeatureAttention(nn.Module):
    # ... setup components ...
    def forward(self, x):
        x_reshaped = x.unsqueeze(-1) # (batch_size, num_features, 1)

        q = self.query_proj(x_reshaped)
        k = self.key_proj(x_reshaped)
        v = self.value_proj(x_reshaped)

        attn_scores = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        attention_output = attn_scores @ v

        output = self.output_proj(attention_output).squeeze(-1)
        return output

```

* **Why it matters:** Standard attention models process sequences of tokens (like words). This module processes **tabular columns** as steps in a sequence.
* **Key Mechanism:** It calculates correlation scores between different statistical features in the dataset. This allows the network to dynamically emphasize relationships between features (e.g., matching a pressure spike with a flow-rate drop) before the data reaches the quantum layers.

---

### 3. Gated Residual Multi-Layer Blocks

```python
x_act = torch.relu(self.linear1(x))
x_act = self.dropout(x_act)
x_act = self.linear2(x_act)
x_act = x_act * torch.sigmoid(self.gate(x_act))
return x_act + residual

```

* **Why it matters:** This block balances deep feature processing with stable gradient flow.
* **Key Mechanism:** The element-wise multiplication by `torch.sigmoid(self.gate(x_act))` acts as an information filter. It lets important feature combinations pass through while blocking noise. The residual addition (`+ residual`) prevents vanishing gradients during backpropagation.

---

### 4. Two-Speed AdamW Parameter Separation

```python
optimizer = optim.AdamW([
    {'params': classical_params, 'lr': 0.005, 'weight_decay': 0.01},
    {'params': quantum_params, 'lr': 0.0005, 'weight_decay': 0.001}
])

```

* **Why it matters:** Classical deep layers and parameter-dependent quantum circuits update at different speeds. Quantum spaces can be highly sensitive to large parameter updates.
* **Key Mechanism:** This block separates the parameters into two optimization tracks. The quantum layers use a learning rate that is 10 times smaller ($5 \times 10^{-4}$) than the classical layers ($5 \times 10^{-3}$). This difference keeps the quantum state transitions stable and prevents the model from suffering from barren plateaus (vanishing quantum gradients) during training.
