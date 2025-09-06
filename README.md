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

- **`lam_pde, lam_phi`** — Weights for PDE residual and $\phi$-regularization terms.
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

  ---
- **Example Case 1:** 
  ```python
  # ==== PDE:  -eps u''(x) + u'(x) = 0 ,  u(0)=0, u(1)=1  ====
  def b_fun(x): return torch.ones_like(x)      # b(x) = +1
  def c_fun(x): return torch.zeros_like(x)     # c(x) = 0
  def f_fun(x): return torch.zeros_like(x)     # f(x) = 0

  # exact solution (stable form; only negative exponentials)
  def exact_solution(x_np, eps):
      """
      Exact for: -eps*u'' + u' = 0,  u(0)=0, u(1)=1.
      u(x) = (e^{x/eps} - 1)/(e^{1/eps} - 1)
         = [ e^{(x-1)/eps} - e^{-1/eps} ] / [ 1 - e^{-1/eps} ]  (stable)
      """
      x = np.asarray(x_np, dtype=np.float64)
      a = 1.0 / eps
      # stable ratio R(x) in [0,1]
      exp_neg_a = np.exp(-a)
      numerator   = np.exp((x - 1.0) * a) - exp_neg_a
      denominator = 1.0 - exp_neg_a
      u = numerator / denominator
      return u.astype(np.float64)

  # ===== Where is the boundary layer?  b>0 ⇒ layer at right (x=1)
  LAYER_SIDE = 'right'

  # ===== Dirichlet–Dirichlet BCs (use hard encoding)
  BC_LEFT  = ('dirichlet', 0.0)   # u(0)=0
  BC_RIGHT = ('dirichlet', 1.0)   # u(1)=1
  BC_MODE  = 'hard'               

  # ===== Epsilon sweep and call
  EPS_LIST = [5e-2, 1e-2, 5e-3, 1e-3]   
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```
  

- **Example Case 2:** 
  ```python
  # PDE:  -ε u'' + u' = 2e^{-x} 
  def b_fun(x): return torch.ones_like(x)
  def c_fun(x): return torch.zeros_like(x)
  def f_fun(x): return 2.0 * torch.exp(-x)

  def exact_solution(x_np, eps):
      """
      Exact for: -eps*u'' + u' = 2*exp(-x),  u(0)=1, u(1)=0
      Numerically stable (only negative exponentials).
      """
      x = np.asarray(x_np, dtype=np.float64)
      a = 1.0/eps
      # stable R(x)
      R = np.exp((x-1.0)*a) * (1.0 - np.exp(-a*x)) / (1.0 - np.exp(-a))
      coef = 2.0/(1.0 + eps)
      u = (1.0 - R) * (1.0 + coef) + R * (coef * np.exp(-1.0)) - coef * np.exp(-x)
      return u.astype(np.float64)

  # Example B: left boundary layer
  LAYER_SIDE = 'right'; BC_LEFT = ('dirichlet', 1.0); BC_RIGHT = ('dirichlet', 0.0)
  EPS_LIST = [5e-2, 1e-2, 5e-3, 1e-3]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```
  
- **Example Case 3:** 
  ```python
  # PDE:  ε y'' + y' + 2e^{-x} = 0   ==>  -ε y'' - y' - 2e^{-x} = 0
  def b_fun(x): return -torch.ones_like(x)        # b = -1
  def c_fun(x): return torch.zeros_like(x)        # c = 0
  def f_fun(x): return 2.0 * torch.exp(-x)        # f = 2e^{-x}

  def exact_solution(x_np, eps):
      """
      Exact solution for: -eps*u'' - u' = k*exp(-x),
      with BCs: u'(0)=0, u(1)=k*exp(-1).
      """
      k=2.0
      x = np.asarray(x_np, dtype=np.float64)
      u = (k / (eps - 1.0)) * (
          eps * (np.exp(-1.0) - np.exp(-1.0/eps))
          + eps * np.exp(-x/eps)
          - np.exp(-x)
      )
      return u.astype(np.float64)


  # Left boundary layer (b<0)
  LAYER_SIDE = 'left'

  # Left Neumann (q0=0), Right Dirichlet (β=2e^{-1})
  BC_LEFT  = ('neumann', 0.0)
  BC_RIGHT = ('dirichlet', float(2.0*np.exp(-1.0)))

  # Your homotopy parameters
  EPS_LIST = [5e-4, 1e-4, 5e-5, 1e-5]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```

- **Example Case 4:** 
  ```python
  # PDE:  ε y'' + y' + e^{-x} = 0   ==>  -ε y'' - y' - e^{-x} = 0
  def b_fun(x): return -torch.ones_like(x)        # b = -1
  def c_fun(x): return torch.zeros_like(x)        # c = 0
  def f_fun(x): return 1.0 * torch.exp(-x)        # f = e^{-x}

  def exact_solution(x_np, eps):
      """
      Exact solution for: -eps*u'' - u' = k*exp(-x),
      with BCs: u'(0)=0, u(1)=k*exp(-1).
      """
      k=1.0
      x = np.asarray(x_np, dtype=np.float64)
      u = (k / (eps - 1.0)) * (
          eps * (np.exp(-1.0) - np.exp(-1.0/eps))
          + eps * np.exp(-x/eps)
          - np.exp(-x)
      )
      return u.astype(np.float64)

  # Left boundary layer (b<0)
  LAYER_SIDE = 'left'

  # Left Neumann (q0=0), Right Dirichlet (β=e^{-1})
  BC_LEFT  = ('neumann', 0.0)
  BC_RIGHT = ('dirichlet', float(1.0*np.exp(-1.0)))

  # Your homotopy parameters
  EPS_LIST = [5e-4, 1e-4, 5e-5, 1e-5]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```

- **Example Case 5:** 
  ```python
  # PDE:  ε y'' + y' + 3e^{-x} = 0   ==>  -ε y'' - y' - 3e^{-x} = 0
  def b_fun(x): return -torch.ones_like(x)        # b = -1
  def c_fun(x): return torch.zeros_like(x)        # c = 0
  def f_fun(x): return 3.0 * torch.exp(-x)        # f = 3e^{-x}

  def exact_solution(x_np, eps):
      """
      Exact solution for: -eps*u'' - u' = k*exp(-x),
      with BCs: u'(0)=0, u(1)=k*exp(-1).
      """
      k=3.0
      x = np.asarray(x_np, dtype=np.float64)
      u = (k / (eps - 1.0)) * (
          eps * (np.exp(-1.0) - np.exp(-1.0/eps))
          + eps * np.exp(-x/eps)
          - np.exp(-x)
      )
      return u.astype(np.float64)

  # Left boundary layer (b<0)
  LAYER_SIDE = 'left'

  # Left Neumann (q0=0), Right Dirichlet (β=3e^{-1})
  BC_LEFT  = ('neumann', 0.0)
  BC_RIGHT = ('dirichlet', float(3.0*np.exp(-1.0)))

  # Your homotopy parameters
  EPS_LIST = [5e-4, 1e-4, 5e-5, 1e-5]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```

- **Example Case 6:** 
  ```python
  # PDE:  ε y'' + y' + 3e^{-x} = 0   ==>  -ε y'' - y' - 4e^{-x} = 0
  def b_fun(x): return -torch.ones_like(x)        # b = -1
  def c_fun(x): return torch.zeros_like(x)        # c = 0
  def f_fun(x): return 4.0 * torch.exp(-x)        # f = 4e^{-x}

  def exact_solution(x_np, eps):
      """
      Exact solution for: -eps*u'' - u' = k*exp(-x),
      with BCs: u'(0)=0, u(1)=k*exp(-1).
      """
      k=3.0
      x = np.asarray(x_np, dtype=np.float64)
      u = (k / (eps - 1.0)) * (
          eps * (np.exp(-1.0) - np.exp(-1.0/eps))
          + eps * np.exp(-x/eps)
          - np.exp(-x)
      )
      return u.astype(np.float64)

  # Left boundary layer (b<0)
  LAYER_SIDE = 'left'

  # Left Neumann (q0=0), Right Dirichlet (β=3e^{-1})
  BC_LEFT  = ('neumann', 0.0)
  BC_RIGHT = ('dirichlet', float(4.0*np.exp(-1.0)))

  # Your homotopy parameters
  EPS_LIST = [5e-4, 1e-4, 5e-5, 1e-5]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```

- **Example Case 7:** 
  ```python
  # PDE:  -ε u'' + u' = e^{-x} 
  def b_fun(x): return torch.ones_like(x)
  def c_fun(x): return torch.zeros_like(x)
  def f_fun(x): return torch.exp(-x)

  def exact_solution(x_np, eps):
      """
      Exact solution for:
          -eps*u'' + u' = k*exp(x),  u(0)=1, u(1)=0.
      Numerically stable (uses only negative exponentials).
      """
      k=1.0
      x = np.asarray(x_np, dtype=np.float64)
      a = 1.0/eps
      # Stable ratio R(x) = (e^{ax}-1)/(e^{a}-1)
      exp_neg_a = np.exp(-a)
      numerator   = np.exp(a*(x-1.0)) * (1.0 - np.exp(-a*x))
      denominator = 1.0 - exp_neg_a
      R = numerator / denominator
      # Closed form
      num = k*np.exp(x) - eps + (-(1.0 - eps) + k*(1.0 - np.e)) * R
      u = num / (1.0 - eps)
      return u.astype(np.float64)

  # Example B: left boundary layer
  LAYER_SIDE = 'right'; BC_LEFT = ('dirichlet', 1.0); BC_RIGHT = ('dirichlet', 0.0)
  EPS_LIST = [5e-2, 1e-2, 5e-3, 1e-3]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```
  
- **Example Case 8:** 
  ```python
  # ==== PDE:  -eps u''(x) + u'(x) = exp(x) ,  u(0)=1, u(1)=0  ====
  def b_fun(x): return torch.ones_like(x)
  def c_fun(x): return torch.zeros_like(x)
  def f_fun(x): return torch.exp(x)

  def exact_solution(x_np, eps):
      import numpy as np
      x = np.asarray(x_np, dtype=np.float64)
      a = 1.0/eps

      # stable ratio R(x) in [0,1]
      # R(x) = (e^{a x}-1)/(e^{a}-1) = e^{a(x-1)} * (1 - e^{-a x})/(1 - e^{-a})
      exp_neg_a = np.exp(-a)
      numerator = np.exp(a*(x-1.0)) * (1.0 - np.exp(-a*x))
      denominator = 1.0 - exp_neg_a
      R = numerator / denominator

      # compact/stable closed form:
      # u(x) = [ e^{x} - ε + (ε - e) * R(x) ] / (1 - ε)
      u = (np.exp(x) - eps + (eps - np.e) * R) / (1.0 - eps)
      return u.astype(np.float64)
    
  LAYER_SIDE = 'right'; BC_LEFT = ('dirichlet', 1.0); BC_RIGHT = ('dirichlet', 0.0)
  EPS_LIST = [5e-2, 1e-2, 5e-3, 1e-3]
  train(EPS_LIST, layer_side=LAYER_SIDE, bc_left=BC_LEFT, bc_right=BC_RIGHT)
  ```

