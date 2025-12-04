#### USMAN AHAD - 2710041

# Knowledge Distillation Techniques: Analysis and Comparison

This report details the implementation and analysis of several Knowledge Distillation (KD) techniques, including standard logit matching, hint-based (feature) transfer, and ensemble methods, evaluated on the CIFAR100 dataset.

## Quantitative Analysis Summary

This table summarizes the final performance and resource profiling metrics for each model.

| Model | Top-1 Accuracy | Top-5 Accuracy | Avg. Latency | Model Size | Energy / Run | MACs / Batch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Independent Student** | 63.76% | 86.43% | 9.06 ms | 37.44 MB | 597.53 mJ | 19705.10 M |
| **Logit Matching (Task 1)** | 67.60% | 88.01% | 9.30 ms | 37.44 MB | 583.08 mJ | 19705.10 M |
| **Hint-Based (Task 3)** | **69.23%** | **88.84%** | 9.23 ms | 37.43 MB | 651.27 mJ | 26952.86 M |
| **Ensemble (Task 4)** | 55.51% | 82.79% | 13.90 ms | 37.06 MB | 960.41 mJ | 19992.67 M |

---

## Task 1: Standard Logit Matching (KD)

### 1.1. Detailed Derivation of $\frac{\partial L_{KD}}{\partial z_i}$

Let $L_{KD} = - \sum_{k} p_k^T \log(p_k^S)$, where $p_k^S = \frac{\exp(z_k/T)}{\sum_j \exp(z_j/T)}$.
The gradient of the softmax is $\frac{\partial p_k^S}{\partial z_i} = \frac{1}{T} p_k^S (\delta_{ik} - p_i^S)$.

$$
\begin{align*}
    \frac{\partial L_{KD}}{\partial z_i}
    &= \frac{\partial}{\partial z_i} \left( - \sum_k p_k^T \log(p_k^S) \right)
    & & \text{\small (Loss function)} \\

    &= - \sum_k p_k^T \frac{1}{p_k^S} \frac{\partial p_k^S}{\partial z_i}
    & & \text{\small (Apply chain rule)} \\

    &= - \sum_k p_k^T \frac{1}{p_k^S} \left( \frac{1}{T} p_k^S (\delta_{ik} - p_i^S) \right)
    & & \text{\small (Substitute softmax gradient)} \\

    &= - \sum_k p_k^T \left( \frac{1}{T} (\delta_{ik} - p_i^S) \right)
    & & \text{\small (Cancel $p_k^S$ terms)} \\

    &= - \frac{1}{T} \sum_k p_k^T (\delta_{ik} - p_i^S)
    & & \text{\small (Factor out constant $\frac{1}{T}$)} \\

    &= - \frac{1}{T} \left( \sum_k p_k^T \delta_{ik} - \sum_k p_k^T p_i^S \right)
    & & \text{\small (Distribute summation)} \\

    &= - \frac{1}{T} \left( p_i^T - p_i^S \sum_k p_k^T \right)
    & & \text{\small (Evaluate first sum, factor $p_i^S$ from second)} \\

    &= - \frac{1}{T} ( p_i^T - p_i^S \cdot 1 )
    & & \text{\small (Since $\sum_k p_k^T = 1$)} \\

    &= - \frac{1}{T} ( p_i^T - p_i^S )
    & & \text{\small (Simplify)} \\

    &= \frac{1}{T} ( p_i^S - p_i^T )
    & & \text{\small (Final form)}
\end{align*}
$$

### 1.2. Training Curves: Independent, Logit Matching, and Hint-Based

These plots compare the training dynamics across the Independent Student, Logit Matching (Task 1), and Hint-Based (Task 3) models.

| Metric | Plot |
| :--- | :--- |
| **Train Loss** | ![Train Loss Comparison](train_loss.png) |
| **Validation Loss** | ![Validation Loss Comparison](val_loss.png) |
| **Train Accuracy** | ![Train Accuracy Comparison](train_acc.png) |
| **Validation Accuracy** | ![Validation Accuracy Comparison](val_acc.png) |

### 1.3. Results: Top-1 / Top-5 Accuracy

| Model | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :---: | :---: |
| **Independent Student** | **63.76%** | **86.43%** |
|
| **KD Student** |  **67.60%** | **88.01%** |


### 1.4. Discussion

It has shown a considerable increase in accuracy (~4% in Top-1 and ~1.5% in Top-5). This is because the student model had considerable more information to work with in the form of the soft-labels that the teacher's logits provide. Normally, one-hot encoded labels (a.k.a hard labels) do not provide any other information, i.e, "This is a cat and this is nothing but a cat." where as soft-labels can provide extra information, i.e, "This is a cat and it is somewhat similar to a dog and is very different from a truck." This extra-information is tapping into the dark knowledge of the teacher and shows a bit of the inter-class relationships that the teacher learns. That is what we're distilling into the student model via soft-labels in the form of teacher logits.

Furthermore, while there wasn't any difference in the convergence behaviour of the independent model and the student (in terms of which epoch they converged at as they both converged at roughly ~40 epochs), the independent model trained twice as fast (owing to the fact that the teacher did not have to incur a forward pass at each batch during training).

---

## Task 2: Analysis of Logit Standardization

### 2.1. Toy Example Results

#### 4-Class Toy Example
* **Teacher Prediction:** Dog
* **Student A Prediction:** Bird (Wrong)
* **Student B Prediction:** Dog (Correct)

| Model | Logits (Original) | KL Divergence (Original) |
| :--- | :--- | :---: |
| Teacher | `[1. 4. 3. 2.]` | - |
| Student A (Wrong) | `[1. 2.8 3. 2.]` | **0.1749** (Undesired lower loss) |
| Student B (Correct)| `[0.1 0.4 0.3 0.2]` | 0.3457 |

| Model | Logits (Standardized) | KL Divergence (Standardized) |
| :--- | :--- | :---: |
| Teacher | `[-1.34 ...]` | - |
| Student A (Wrong) | `[-1.52 ...]` | 0.1347 |
| Student B (Correct)| `[-1.34 ...]` | **0.0000** (Desired lower loss) |

#### 10-Class Toy Example
* **Teacher Prediction:** deer
* **Student A Prediction:** truck (Wrong)
* **Student B Prediction:** deer (Correct)

| Model | Logits (Original) | KL Divergence (Original) |
| :--- | :--- | :---: |
| Teacher | `[1. 4. ... 4.5]` | - |
| Student A (Wrong) | `[1.1 4.2 ... 7.5]` | **0.7192** (Undesired lower loss) |
| Student B (Correct)| `[0.1 0.4 ... 0.45]` | 1.2317 |

| Model | Logits (Standardized) | KL Divergence (Standardized) |
| :--- | :--- | :---: |
| Teacher | `[-1.07 ... 0.85]` | - |
| Student A (Wrong) | `[-1.02 ... 1.91]` | 0.1923 |
| Student B (Correct)| `[-1.07 ... 0.85]` | **0.0000** (Desired lower loss) |

### 2.2. CIFAR100 Mismatch Cases

There were **43** total mismatches across the training and test sets during the evalutation of the models of which only **6** were fixed by the standardization. The following is one sample of such a fix.

### 2.3. Summary Table

| Setup | KL Divergence (ORIGINAL) | KL Divergence (STANDARDIZED) | Ranking Correlation (Spearman) | Prediction Correctness |
| :--- | :---: | :---: | :---: | :---: |
| **Independent Model** | 0.9000 | 0.3360 | 0.7063 | CORRECT |
| **Distilled Model**| 0.7206 | 0.4629 | 0.5695 | WRONG |

### 2.4. Discussion

The toy examples clearly indicate an unfairly high loss for the correct student as compared to the incorrect student. This is because the incorrect student was only "slightly" wrong but its distribution almost directly matched the teacher's.  By scaling all logits to a standard distribution, tandardization removes the influence of magnitude and prevents the teacher's "confidence" from dominating the loss signal. This leads to a more stable and reliable gradient. With magnitude removed, the loss function is forced to only evaluate the relative ranking and shape of the student's distribution compared to the teacher's. This is the true goal of "dark knowledge" transfer. As seen in the toy example, after standardization, the correct student got its loss reduced to zero (which is what it should be) and the incorrect student got penalised more.


## Task 3: Hint-Based (Feature) Distillation

### 3.1. Training/Validation Curves (Comparison with Logit Matching)

**NOte:** See above

### 3.2. Quantitative Comparison Table

| Metric | Logit Matching (Task 1) | Hint-Based (Task 3) |
| :--- | :---: | :---: |
| **Top-1 Accuracy** | 67.60% | **69.23%** |
| **Top-5 Accuracy** | 88.01% | **88.84%** |
| **Avg. Latency** | 9.30 ms | 9.23 ms |
| **Energy / Run** | 583.08 mJ | 651.27 mJ |
| **MACs / Batch** | 19705.10 M | **26952.86 M** |
| **Convergence Speed** | **~40 epochs** | ~40 epochs |

**Note:** Logit Matching Epochs were half as time-consuming than Hint-Based Epochs

### 3.3. Discussion

Hint-Losses serve as a way to directly enforce what the student is learning and HOW its learning by directly trying to match the activations of the student and teacher up to the hints layer. This forces the student to replicate the teacher's intermediary representations leading to a much better accuracy (the best by a considerable margin at around ~1.5% better than logit matching). This higher performance however came at the cost of much longer memory overhead and performance requirements and a much longer training time despite it still converging at around the same point (but each epoch took twice as long as logit matching which was twice as long as an independent model, meaning that it took a hit of a 4x increase in training time!)


## Task 4: Ensemble of Sub-Students

### 4.1. Training Logs and Validation Plots

| Metric | Plot |
| :--- | :--- |
| **Train Loss (Individual vs. Ensemble)** | ![Ensemble Train Loss](ensemble.png) |
| **Validation Loss & Accuracy (Individual vs. Ensemble)** | ![Ensemble Val Loss Acc](ensemble_Acc.png) |

### 4.2. Comparison Table

| Model | Parameter Count (Size) | Top-1 Accuracy | Top-5 Accuracy |
| :--- | :---: | :---: | :---: |
| **Single Student (Indep, T1)** | 37.44 MB | 63.76% | 86.43% |
| **Single Student (LM, T1)** | 37.44 MB | 67.60% | 88.01% |
| **Single Student (Hint, T3)** | 37.43 MB | **69.23%** | **88.84%** |
| **Ensembled Model (Combined)** | 37.06 MB | 55.51% | 82.79% |

### 4.3. Discussion

This model performed the worse by far. It could be due to several reasons:
- In trying to make things fair, I tried to keep the number of epochs for each student model lower (25 compared to 50) as this model was roughly 1/4th the size. Maybe increasing the epochs would result in far better accuracy.
- If not that, then the number of parameters is definitely the issue as 2.5 million parameters is far too little to incur a substantial accuracy output, especially for a difficult dataset like CIFAR-100.

Moreover, the ensemble had the highest latency and Energy per Run by far (~1.5x that of the others) due to the number of forward passes (4 instead of 1), but interestingly it had the least amount of memory consumption, possibly due to the smaller models requiring less overhead. It did however turn out to be roughly the same size (2407680 which is roughly 1/4th of a standard VGG-11 Model). However, this minor benefit is far outweighed by its high latency and extremely poor accuracy, making this approach computationally inefficient and ineffective.