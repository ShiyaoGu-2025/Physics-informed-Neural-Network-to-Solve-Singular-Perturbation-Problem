# Physics-informed-Neural-Network-to-Solve-Singular-Perturbation-Problem
Author: Shiyao Gu; Jierui Li

# Hyperparameter Documentation


---

## 1) Boundary Layer and Boundary Conditions

- **`LAYER_SIDE`** — Side where the boundary layer is expected.  
  - Options: `'left'` | `'right'`  
  - Example:
    ```python
    LAYER_SIDE = 'right'
    ```

- **`BC_LEFT`, `BC_RIGHT`** — Boundary conditions at \(x=0\) and \(x=1\).  
  - Format:
    - `('dirichlet', α)`  → Dirichlet BC \(u = $\alpha$\)
    - `('neumann', q)`   → Neumann BC \(u' = q\)
  - Example:
    ```python
    BC_LEFT  = ('dirichlet', 0.0)
    BC_RIGHT = ('dirichlet', 1.0)
    ```

---

## 2) Training Parameters

- **`EPS_LIST`** — List of $\epsilon$ values to sweep.  
  - Example:
    ```python
    EPS_LIST = [5e-2, 1e-2, 5e-3, 1e-3]
    ```

- **`NCOL`, `NEDGE`** — Interior collocation points vs. boundary/edge-focused points.  
  - Example:
    ```python
    NCOL, NEDGE = 2048, 512
    ```

- **`EPOCHS_PER_STAGE`** — Epochs per training stage (e.g., per-ε schedule).
  ```python
  EPOCHS_PER_STAGE = 2000
  ```

- **`PRINT_EVERY`, `PLOT_EVERY`** — Logging and visualization frequency (in steps).
  ```python
  PRINT_EVERY = 100
  PLOT_EVERY  = 500
  ```

- **`LR`** — Optimizer learning rate.
  ```python
  LR = 1e-3
  ```

- **`BC_MODE`** — How boundary conditions are imposed.  
  - `'hard'`: encode BCs into the solution ansatz.  
  - `'soft'`: add a boundary penalty term to the loss.
  ```python
  BC_MODE = 'hard'
  ```

- **`LAM_BC`** — Boundary penalty weight used **only when** `BC_MODE = 'soft'`.
  ```python
  LAM_BC = 1.0
  ```

---

## 3) Branch-Gated / φ Regularization

- **`S0, S1, kappa`** — Region-splitting / gating parameters (inner / transition / outer).
  ```python
  S0, S1, kappa = 5.0, 7.0, 2.0
  ```

- **`lam_comp`** — Compatibility/consistency penalty of transition band.
  ```python
  lam_comp = 0.05
  ```

- **`tau, eta_phi`** — Regularization for the learned coordinate transform $\phi(x)$
  ```python
  tau, eta_phi = 0.1, 1e-4
  ```

- **`lam_pde, lam_phi`** — Weights for PDE residual and \(\phi\)-regularization terms.
  ```python
  lam_pde, lam_phi = 1.0, 0.01
  ```

---

## 4) PDE Coefficients and (Optional) Exact Solution

- **General PDE form**
  
  $- \varepsilon u''(x) + b(x) u'(x) + c(x) u(x) = f(x)$
  

- **Coefficient stubs** — provide your problem-specific definitions:
  ```python
  def b_fun(x): ...
  def c_fun(x): ...
  def f_fun(x): ...
  ```

- **Exact solution (optional)** — if an analytical solution exists for your chosen
  \(b(x), c(x), f(x)\) and BCs, you can implement it for visualization/validation:
  ```python
  def exact_solution(x_np, eps):
      # return a NumPy array with u(x) for plotting / error metrics
  ```
