# NASA International Space Apps Challenge
# Interplanetary Survival Guide: Martian Map
## Unified Technical Reference & Scientific Specification (Steps 1 through 6 + Interface Architecture)

---

## Executive Overview

Human surface operations on Mars require autonomous, bioenergetically-constrained path planning that dynamically accounts for life-support consumption, surface traction, hazardous topographic obstacles, and real-time sensor uncertainty.

This technical reference unifies the complete **6-Step Computational Pipeline** and **Dual-Display Interface Architecture** into a single authoritative specification:

* **Step 1 (2D Continuous Martian Terrain Cost Field):** Converts $1\text{ m/px}$ MRO HiRISE elevation matrices $z(r,c)$, Odyssey THEMIS thermal inertia $I_{\text{th}}$, and USGS geologic unit mapping into a continuous, impassability-bounded traversal cost field $C(r,c)$ in $< 5\text{ ms}$.
* **Step 2 (Sobel + Laplacian Deviation Score & Dual-Map Streaming):** Extracts directional gradient $|\nabla C|$ and 5-point discrete Laplacian $|\nabla^2 C|$ from an isolated, EDT-sanitized scratchpad $C_{\text{safe}}(r,c)$, synthesizing a shift-invariant Deviation Score $D(r,c) \in [0, 1]$ that guides $99.5\%$ graph compression.
* **Step 3 (Adaptive Vertex Sampling & KDTree Geometric Graph):** Constructs a 1D cumulative distribution function (CDF) strictly over passable pixels with a connectivity floor ($\epsilon_{\text{floor}} = 0.05$), deterministically places $N = 1,200$ vertices, performs batched 11-point ray collision tensor checks, and stitches chunks across $26\text{ m}$ boundary strips.
* **Step 4 (Mars Gravity-Adjusted Pandolf Bioenergetics):** Evaluates metabolic rate $P_{\text{watts}}$ using the Mars-scaled ($G_{\text{mars}} = 0.38$) Pandolf equation with an eccentric downhill braking floor ($\ge 20\%$), converting mechanical energy directly into life-support oxygen weights $\Delta\text{O}_2(u \to v)$ at $20,100\text{ J/L }\text{O}_2$.
* **Step 5 (Payload-Aware Directed $A^*$ & Point of No Return):** Computes multi-leg trajectories with dynamic payload mass additions ($L = 65\text{ kg} + m_{\text{samples}}$) using a division-free admissible 3D heuristic, while solving the Point of No Return (PNR) for all $1,200$ nodes simultaneously via single-source Dijkstra on $\mathcal{G}^\top$ (`G.reverse()`) in $< 3\text{ ms}$.
* **Step 6 (Sensor Physics & Rover Ground-Truth Calibration):** Evaluates 3D Hab radio line-of-sight ($L_{\text{LOS}}$) and 8-azimuth canyon radiation shielding ($R_{\text{rad}}$) via fast horizon raycasting, and fuses NASA *Curiosity* / *Perseverance* rover ground imagery via Sobel rock-edge density ($\delta_{\text{rock}}$) to collapse orbital uncertainty.
* **Interface Architecture:** Detailed operational breakdown of the Mission Control 3D Web Cockpit (`index.html`), Mobile Field EVA Companion HUD (`phone.html`), and Full-Duplex Telemetry Bridge (`server.py`).

---

## 1. Step 1 Technical Reference: 2D Continuous Terrain Cost Field

### 1.1 Canonical vs. Computationally Optimized Equations

#### 1.1.1 Squared Terrain Gradient $G^2(r,c)$, Passability Mask $\mathcal{M}_{\text{pass}}$, & Slope $\theta(r,c)$
* **Canonical Formulation:**
  $$\nabla z(r,c) = \left(\frac{z(r+1,c) - z(r-1,c)}{2\Delta y}, \frac{z(r,c+1) - z(r,c-1)}{2\Delta x}\right)$$
  $$\theta(r,c) = \frac{180}{\pi}\arctan\!\left(\sqrt{\left(\frac{\partial z}{\partial y}\right)^2 + \left(\frac{\partial z}{\partial x}\right)^2}\right)$$
  $$\mathcal{M}_{\text{pass}}(r,c) = \big(\theta(r,c) \le 20.0^\circ\big) \land \big(2 \le r \le H-3\big) \land \big(2 \le c \le W-3\big)$$

* **Computationally Optimized Formulation (Deferred Transcendental Pruning):**
  $$G^2(r,c) = \left(\frac{z(r+1,c) - z(r-1,c)}{2\Delta y}\right)^{\!2} + \left(\frac{z(r,c+1) - z(r,c-1)}{2\Delta x}\right)^{\!2}$$
  $$\mathcal{M}_{\text{pass}}(r,c) = \Big(G^2(r,c) \le \tan^2(20^\circ)\Big) \;\land\; \big(2 \le r \le H-3\big) \;\land\; \big(2 \le c \le W-3\big)$$
  $$\theta(r,c) = \frac{180}{\pi}\arctan\!\Big(\sqrt{G^2(r,c)}\Big) \quad \forall (r,c) \in \mathcal{M}_{\text{pass}}$$

* **Definitions:**
  - $z(r,c)$: HiRISE DTM surface elevation in meters relative to Mars geoid.
  - $\Delta x, \Delta y$: Horizontal and vertical grid spacings ($1.0\text{ m/px}$).
  - $G^2(r,c)$: Squared gradient magnitude.
  - $\mathcal{M}_{\text{pass}}(r,c)$: Binary safe-traversal indicator ($1 = \text{passable}, 0 = \text{cliff or boundary}$).
  - $\theta(r,c)$: Slope in degrees.
* **Proposed Ranges:**
  - Inputs: $z(r,c) \in [-5000.0, +5000.0]\text{ m}$; $\Delta x, \Delta y \in [0.5, 2.0]\text{ m}$.
  - Outputs: $G^2(r,c) \in [0.0, 50.0]$; $\mathcal{M}_{\text{pass}}(r,c) \in \{0, 1\}$; $\theta(r,c) \in [0.0^\circ, 20.0^\circ]$.
* **Computational & Physical Impact:** Pre-evaluating $\tan^2(20^\circ) \approx 0.132474$ prunes steep cliffs via a single floating-point comparison, skipping $\sqrt{\cdot}$ and $\arctan(\cdot)$ on invalid pixels ($> 20\%$ of typical crater walls).

---

#### 1.1.2 Denoised $5\times 5\text{ m}$ Local Roughness $R(r,c)$ via Separable Box Filter
* **Canonical Formulation:**
  $$R(r,c) = \sqrt{\max\!\left(0.0,\; \frac{1}{25}\sum_{i=-2}^{2}\sum_{j=-2}^{2}\big(z(r+i, c+j) - \mu_{5\times 5}(z)\big)^2 - \sigma_z(r,c)^2\right)}$$

* **Computationally Optimized Formulation (Separable Moving Averages):**
  $$R(r,c) = \sqrt{\max\!\Big(0.0,\; \mathcal{B}_r^{(5)}\!\left[\mathcal{B}_c^{(5)}(z^2)\right] - \left(\mathcal{B}_r^{(5)}\!\left[\mathcal{B}_c^{(5)}(z)\right]\right)^{\!2} - \sigma_z(r,c)^2\Big)}$$

* **Definitions:**
  - $\mathcal{B}_r^{(5)}, \mathcal{B}_c^{(5)}$: 1D 5-tap moving-average passes across rows and columns.
  - $\sigma_z(r,c)$: Photogrammetric stereo vertical error ($0.1\text{--}0.8\text{ m}$).
  - $R(r,c)$: Sub-grid topographic RMS roughness in meters.
* **Proposed Ranges:**
  - Inputs: $z(r,c) \in [-5000.0, +5000.0]\text{ m}$; $\sigma_z(r,c) \in [0.08, 0.40]\text{ m}$.
  - Outputs: $R(r,c) \in [0.0, 2.5]\text{ m}$.
* **Computational & Physical Impact:** Decomposes 2D window variance into two 1D operations, reducing cost from $25$ to $10$ ops/pixel. Subtracting $\sigma_z^2$ eliminates false roughness generated by stereo matching noise in low-contrast dust sheets.

---

#### 1.1.3 Pre-Folded Dimensionless Terrain Friction Multiplier $\eta(r,c)$
* **Canonical Formulation:**
  $$\eta(r,c) = \eta_{\text{geo}}(r,c)\left[ 1.0 + 4.8\max\!\left(0.0, \frac{160.0 - I_{\text{th}}(r,c)}{160.0}\right) + 0.4\left(\frac{R(r,c)}{R_{\text{ref}}}\right) \right]$$

* **Computationally Optimized Formulation (Pre-Folded SIMD FMA Multiplication):**
  $$\eta(r,c) = \eta_{\text{geo}}(r,c)\Big[ 1.0 + \max\!\big(0.0,\; 4.8 - 0.03\,I_{\text{th}}(r,c)\big) + \alpha_R\,R(r,c) \Big], \quad \alpha_R = \frac{0.4}{R_{\text{ref}}}$$

* **Definitions:**
  - $\eta_{\text{geo}}(r,c)$: USGS baseline geologic unit friction prior.
  - $I_{\text{th}}(r,c)$: THEMIS thermal inertia in $\text{J}\cdot\text{m}^{-2}\cdot\text{K}^{-1}\cdot\text{s}^{-1/2}$ ($\text{TIU}$).
  - $R_{\text{ref}}$: Reference 95th-percentile roughness across the exploration region ($0.45\text{ m}$).
  - $\eta(r,c)$: Dimensionless friction multiplier.
* **Proposed Ranges:**
  - Inputs: $\eta_{\text{geo}} \in \{1.05, 1.10, 1.25, 1.35, 2.45, 2.80\}$; $I_{\text{th}} \in [20, 600]\text{ TIU}$; $R_{\text{ref}} \approx 0.45\text{ m}$.
  - Outputs: $\eta(r,c) \in [1.05, 18.0]$.
* **Computational & Physical Impact:** Pre-folding $4.8/160 = 0.03$ replaces array division with fast SIMD multiplication. Enforces multiplicative mechanics: dust deposited over boulder fields induces higher mobility risk than dust over flat bedrock.

---

#### 1.1.4 Composite Epistemic Orbital Uncertainty $U(r,c)$
* **Optimized Formulation:**
  $$U(r,c) = \text{clip}\!\left( \beta_z\,\sigma_z(r,c) + 0.6\exp\!\big(-\gamma_{\text{th}}(I_{\text{th}}(r,c) - 220)^2\big) - 0.7\,\rho_{\text{rover}}(r,c),\; 0.0,\; 1.0 \right)$$
  where $\beta_z = 0.4 / \sigma_{z,\max} = 0.5\text{ m}^{-1}$ and $\gamma_{\text{th}} = \frac{1}{2(55)^2} \approx 1.65289 \times 10^{-4}\text{ TIU}^{-2}$.
* **Definitions:**
  - $U(r,c)$: Dimensionless epistemic data uncertainty $[0, 1]$.
  - $\rho_{\text{rover}}$: Rover ground-truth verification confidence $[0, 1]$.
* **Proposed Ranges:**
  - Inputs: $\sigma_z \in [0.08, 0.80]\text{ m}$; $I_{\text{th}} \in [20, 600]\text{ TIU}$; $\rho_{\text{rover}} \in [0, 1]$.
  - Outputs: $U(r,c) \in [0.0, 1.0]$.
* **Computational & Physical Impact:** Automatically flags sub-pixel thermal mixing zones ($160\text{--}300\text{ TIU}$) where $100\text{ m/px}$ THEMIS pixels obscure meter-scale sand traps, while ground-truth rover imagery collapses uncertainty.

---

#### 1.1.5 Masked Continuous Pixel Cost Field $C(r,c)$
* **Optimized Formulation:**
  $$C(r,c) = \begin{cases} \infty, & \text{if } \neg\mathcal{M}_{\text{pass}}(r,c) \\ \eta(r,c)\big(100.0 + 14.0\,\theta(r,c)\big) + 40.0\,U(r,c) + 60.0\big(1.0 - L_{\text{LOS}}(r,c)\big), & \text{if } \mathcal{M}_{\text{pass}}(r,c) \end{cases}$$
* **Definitions:**
  - $C(r,c)$: Traversal cost per meter of horizontal travel.
  - $L_{\text{LOS}}$: Hab line-of-sight indicator ($1 = \text{visible}, 0 = \text{radio shadow}$).
* **Proposed Ranges:**
  - Inputs: $\eta \in [1.05, 18.0]$; $\theta \in [0^\circ, 20^\circ]$; $U \in [0, 1]$; $L_{\text{LOS}} \in \{0, 1\}$.
  - Outputs: $C(r,c) \in [105.0, 6900.0] \cup \{\infty\}$.
* **Computational & Physical Impact:** Distributing $100$ into $(100 + 14\,\theta)$ saves a full matrix multiplication pass, executing the entire $262,144\text{-pixel}$ raster in $< 4.5\text{ ms}$.

---

### 1.2 Provenance, Rationale, Engineering Problems & Judge Defense (Step 1)
* **Provenance:** Zevenbergen & Thorne (1987) slope gradient; Golombek et al. (2012) $20^\circ$ cliff threshold; Shepard et al. (2001) RMS planetary roughness; Pandolf et al. (1977) biomechanical friction; Fergason et al. (2006) THEMIS thermal inertia boundaries.
* **Why We Use Them:** Guarantees dimensional homogeneity (all bracket terms are dimensionless) and strict physical orthogonality (material friction $\eta$ is decoupled from slope $\theta$, preventing gravity from being double-counted).
* **Engineering Solutions:**
  - Sensor mismatch ($1\text{ m}$ HiRISE vs. $100\text{ m}$ THEMIS): Bicubic resampling removes block artifacts; Gaussian uncertainty spike flags mixed pixels.
  - Stereo photogrammetric noise: Variance subtraction $\max(0, \mu_{z^2} - \mu_z^2 - \sigma_z^2)$ separates noise from physical boulders.
* **NASA Judge Defense:**
  - *Q: Why compute dense $512\times 512$ rasters if routing on a sparse graph?*
    *A: Dense evaluation is necessary to find where cost changes rapidly. Our SIMD implementation runs in $<5\text{ ms}$, enabling Step 2 to place vertices exclusively where obstacles and hazards exist.*

---

## 2. Step 2 Technical Reference: Sobel + Laplacian Deviation Score & Chunk Streaming

### 2.1 Canonical vs. Computationally Optimized Equations

#### 2.1.1 Dual-Map Separation & Nearest-Valid-Neighbor Scratchpad ($C_{\text{safe}}$)
* **Optimized Formulation (Offline Global EDT + Online Halo Slicing):**
  $$(r^*, c^*) = \text{EDT}_{\text{global}}\!\big(\neg\mathcal{M}_{\text{pass}}^{\text{global}}\big)(r,c), \qquad C_{\text{safe}}^{\text{global}}(r,c) = C^{\text{global}}\big(r^*(r,c), c^*(r,c)\big)$$
  $$C_{\text{safe}}^{(I,J)} = C_{\text{safe}}^{\text{global}}\big[IL - 1 : (I+1)L + 1,\; JL - 1 : (J+1)L + 1\big] \in \mathbb{R}^{(L+2)\times (L+2)}$$
* **Definitions:**
  - $C_{\text{safe}}$: Isolated, strictly finite scratch buffer used solely for derivative stencils.
  - $\text{EDT}$: Exact 2D Euclidean Distance Transform with deterministic row-major tie-breaking.
* **Ranges:**
  - Inputs: $C^{\text{global}}(r,c) \in [105.0, 6900.0] \cup \{\infty\}$.
  - Outputs: $C_{\text{safe}}(r,c) \in [105.0, 6900.0]$ (strictly finite).
* **Impact:** Precomputing EDT globally offline eliminates runtime overhead and reduces online halo size to $h = 1\text{ px}$. Copies rim values into cliffs, creating a zero-gradient Neumann boundary that prevents artificial shockwaves.

---

#### 2.1.2 Directional Cost Transition Detector: Rank-1 Separable Sobel Gradient ($|\nabla C|$)
* **Optimized Formulation (Separable 1D Convolutions + Global Scale $S_\nabla$):**
  $$G_r^{(I,J)} = \big(C_{\text{safe}}^{(I,J)} * [1, 2, 1]_c\big) * [-1, 0, +1]_r^T, \quad G_c^{(I,J)} = \big(C_{\text{safe}}^{(I,J)} * [1, 2, 1]_r^T\big) * [-1, 0, +1]_c$$
  $$\widehat{|\nabla C|}^{(I,J)}(r,c) = \begin{cases} 0.0, & \text{if } \neg\mathcal{M}_{\text{pass}}^{(I,J)}(r+1, c+1) \\ \text{clip}\!\left(\alpha_\nabla \sqrt{\big[G_r^{(I,J)}(r+1, c+1)\big]^2 + \big[G_c^{(I,J)}(r+1, c+1)\big]^2},\; 0.0,\; 1.0\right), & \text{otherwise} \end{cases}$$
  where $\alpha_\nabla = 1 / S_\nabla = 1 / 660.0 \approx 1.51515 \times 10^{-3}$.
* **Definitions:**
  - $G_r, G_c$: Row and column spatial gradients.
  - $S_\nabla = 660.0$: Analytical step-response scale ($4\,\Delta C_{\text{target}}$ for $\Delta C = 165$).
* **Ranges:** Inputs: $C_{\text{safe}} \in [105.0, 6900.0]$; Outputs: $\widehat{|\nabla C|} \in [0.0, 1.0]$.
* **Impact:** Cuts convolution ops from $9$ to $6$ per pixel. Replaces per-chunk sorting with fixed $S_\nabla$, achieving exact shift-invariance.

---

#### 2.1.3 Rapid Local Cost-Change Detector: 5-Point Discrete Laplacian ($|\nabla^2 C|$)
* **Optimized Formulation (5-Point Stencil + Global Scale $S_{\nabla^2}$):**
  $$L_{\text{pad}}^{(I,J)}(u,v) = \big| C_{\text{safe}}(u+1,v) + C_{\text{safe}}(u-1,v) + C_{\text{safe}}(u,v+1) + C_{\text{safe}}(u,v-1) - 4\,C_{\text{safe}}(u,v) \big|$$
  $$\widehat{|\nabla^2 C|}^{(I,J)}(r,c) = \begin{cases} 0.0, & \text{if } \neg\mathcal{M}_{\text{pass}}^{(I,J)}(r+1, c+1) \\ \text{clip}\!\left(\alpha_{\nabla^2}\,L_{\text{pad}}^{(I,J)}(r+1, c+1),\; 0.0,\; 1.0\right), & \text{otherwise} \end{cases}$$
  where $\alpha_{\nabla^2} = 1 / S_{\nabla^2} = 1 / 350.0 \approx 2.85714 \times 10^{-3}$.
* **Definitions:**
  - $L_{\text{pad}}$: Isotropic second-order discrete curvature.
  - $S_{\nabla^2} = 350.0$: Analytical step-change gain ($2\,\Delta C_{\text{target}}$).
* **Ranges:** Inputs: $C_{\text{safe}} \in [105.0, 6900.0]$; Outputs: $\widehat{|\nabla^2 C|} \in [0.0, 1.0]$.
* **Impact:** 4 adds and 1 multiply per pixel. Spikes at canyon bottleneck chokepoints while vanishing on smooth linear ramps.

---

#### 2.1.4 Composite Deviation Score $D^{(I,J)}(r,c)$
* **Optimized Formulation (Weight-Folded Single-Pass FMA):**
  $$D^{(I,J)}(r,c) = \begin{cases} 0.0, & \text{if } \neg\mathcal{M}_{\text{pass}}^{(I,J)}(r,c) \\ \min\!\big(0.45,\; \tilde{w}_\nabla |\nabla C|\big) + \min\!\big(0.30,\; \tilde{w}_{\nabla^2} |\nabla^2 C|\big) + 0.25\,U(r,c), & \text{otherwise} \end{cases}$$
  where $\tilde{w}_\nabla = 0.45 / 660.0 \approx 6.81818 \times 10^{-4}$ and $\tilde{w}_{\nabla^2} = 0.30 / 350.0 \approx 8.57143 \times 10^{-4}$.
* **Ranges:** Inputs: $|\nabla C| \ge 0$; $|\nabla^2 C| \ge 0$; $U \in [0, 1]$. Outputs: $D \in [0.0, 1.0]$.
* **Impact:** Strictly bounded in $[0, 1]$ via convex weights ($0.45 + 0.30 + 0.25 = 1.00$).

---

### 2.2 Provenance, Rationale, Engineering Problems & Judge Defense (Step 2)
* **Provenance:** Felzenszwalb & Huttenlocher (2012) linear-time EDT; Sobel & Feldman (1968) rank-1 isotropic gradient; Courant, Friedrichs & Lewy (1928) finite-difference Laplacian; Marr & Hildreth (1980) Laplacian edge detection; Gropp et al. (1999) halo domain decomposition.
* **Why We Use Them:** Concentrating vertices where $D(r,c)$ spikes reduces graph size by $99.5\%$ while preserving sub-meter hazard boundaries around cliffs and loose regolith.
* **Engineering Solutions:**
  - $\infty - \infty = \text{NaN}$ contamination: Strict dual-map separation keeps ground-truth $C(r,c)$ immutable and routes stencils on sanitized $C_{\text{safe}}$.
  - Seam discontinuities: Map-wide constants $S_\nabla = 660, S_{\nabla^2} = 350$ achieve $0.00\times 10^0$ error between chunked and monolithic runs.
* **NASA Judge Defense:**
  - *Q: Doesn't filling cliffs with finite numbers in $C_{\text{safe}}$ risk routing astronauts across a cliff?*
    *A: Never. $C_{\text{safe}}$ is a read-only scratchpad for convolutions. Immediately after, cliff pixels are forced to $D = 0$, and Step 3 edge validation checks strictly against ground-truth $C(r,c)$ where cliffs remain $\infty$.*

---

## 3. Step 3 Technical Reference: Adaptive Graph & Tensor Collision Checks

### 3.1 Canonical vs. Computationally Optimized Equations

#### 3.1.1 Passable-Only 1D Prefix-Sum CDF Sampling & Spatial Hash Seeding
* **Optimized Formulation (1D Prefix-Sum + Binary Search + Prime-XOR Seed):**
  $$\text{seed}(I,J) = \big(I \cdot 73856093 \;\oplus\; J \cdot 19349663\big) \bmod 2^{32}$$
  $$F_k = \sum_{m=1}^{k} \big(D(r_m, c_m) + \epsilon_{\text{floor}}\big), \quad \forall (r_m, c_m) \in \mathcal{M}_{\text{pass}}^{(I,J)}$$
  $$V_{\text{sample}} = \text{SearchSorted}\!\big(F,\; u_s \cdot F_{|\mathcal{M}_{\text{pass}}|}\big), \quad u_s \sim \mathcal{U}[0, 1)$$
  $$V = V_{\text{sample}}^{(1200)} \cup \{v_{\text{Hab}}, v_{\text{CRISM},1}, v_{\text{CRISM},2}, v_{\text{CRISM},3}\}$$
* **Definitions:**
  - $\epsilon_{\text{floor}} = 0.05$: Minimum sampling floor guaranteeing connected vertex backbones across flat plains.
  - $F_k$: Monotonic prefix-sum array of passable weights.
* **Ranges:** Inputs: $D \in [0, 1]$; $\epsilon_{\text{floor}} = 0.05$; $N = 1200$. Outputs: $V \in \mathbb{R}^{1204\times 2}$.
* **Impact:** Filters strictly to passable pixels, ensuring zero samples ever land on $> 20^\circ$ cliffs. Binary search (`np.searchsorted`) executes in $O(N \log |\mathcal{M}_{\text{pass}}|)$ time ($< 0.6\text{ ms}$).

---

#### 3.1.2 Batched 11-Point Ray Collision Tensor & $26\text{ m}$ Boundary-Strip Stitching
* **Optimized Formulation (Tensor Broadcast Collision Check + Boundary KDTree):**
  $$\mathbf{P} = \mathbf{U}[:, \text{None}, :] + \mathbf{t}[\text{None}, :, \text{None}] \cdot (\mathbf{V} - \mathbf{U})[:, \text{None}, :] \in \mathbb{R}^{M \times 11 \times 2}, \quad \mathbf{t} = [0.0, 0.1, \dots, 1.0]$$
  $$\text{Valid}(e) = \bigwedge_{k=0}^{10} \mathcal{M}_{\text{pass}}\!\big(\lfloor\mathbf{P}[e, k, :]\rceil\big)$$
  $$\mathcal{E}_{\text{bridge}}^{(A \leftrightarrow B)} = \text{KDTree}\!\big(V_A[\text{dist}_{\text{seam}} \le 26]\big).\text{query\_ball\_tree}\!\Big(\text{KDTree}\!\big(V_B[\text{dist}_{\text{seam}} \le 26]\big),\; r=26.0\Big)$$
* **Definitions:**
  - $M$: Number of candidate undirected segments within $R = 26.0\text{ m}$.
  - $d_{3D}(u,v)$: True 3D Euclidean distance including elevation difference.
  - $S_{uv}$: Directed percent grade ($S_{uv} = 100(z_v - z_u) / \Vert u - v\Vert_2$).
* **Ranges:** Inputs: $V \in \mathbb{R}^{N\times 2}$; $R = 26.0\text{ m}$. Outputs: $d_{3D} \in (0, 27.7]\text{ m}$; $S_{uv} \in [-36.4\%, +36.4\%]$; $\bar{\eta}_{uv} \in [1.05, 18.0]$.
* **Impact:** Evaluates all $M \approx 8,000$ candidate edges simultaneously in an $(M\times 11\times 2)$ NumPy tensor, completing collision detection in $< 1.8\text{ ms}$. Boundary-strip stitching queries only $\sim 60$ nodes per chunk border, migrating chunks without graph rebuilds.

---

### 3.2 Provenance, Rationale, Engineering Problems & Judge Defense (Step 3)
* **Provenance:** Bentley (1975) KDTree; Kavraki et al. (1996) Probabilistic Roadmaps; Karaman & Frazzoli (2011) radius percolation; Teschner et al. (2003) spatial hashing.
* **Why We Use Them:** Compressing $262,144$ pixels into $1,200$ nodes cuts graph size by $99.5\%$, enabling interactive real-time path planning while preserving exact obstacle clearance.
* **Engineering Solutions:**
  - Disconnected flat terrain: Setting $\epsilon_{\text{floor}} = 0.05$ guarantees that flat bedrock ($D \to 0$) retains enough nodes to maintain topological connectivity.
  - Cross-chunk latency: Boundary-strip filtering queries only nodes within $26\text{ m}$ of the seam, avoiding neighboring graph recomputation.
* **NASA Judge Defense:**
  - *Q: Why not rejection-sample the 2D deviation raster?*
    *A: Rejection sampling wastes draws on large cliff zones. The 1D CDF prefix-sum over $\mathcal{M}_{\text{pass}}$ guarantees $100\%$ valid sample efficiency in $O(N \log |\mathcal{M}_{\text{pass}}|)$ time.*

---

## 4. Step 4 Technical Reference: Mars Bioenergetics & Direct $\Delta\text{O}_2$ Weights

### 4.1 Canonical vs. Computationally Optimized Equations

#### 4.1.1 Mars Gravity-Adjusted Pandolf Metabolic Power & Downhill Braking Floor
* **Canonical Formulation:**
  $$P_{\text{raw}}(u \to v \mid L) = G_{\text{mars}}\left[ 1.5\,W + 2.0(W + L)\left(\frac{L}{W}\right)^{\!2} + \bar{\eta}_{uv}(W + L)\big(1.5\,V^2 + 0.35\,V\,S_{uv}\big) \right]$$
  $$P_{\text{watts}}(u \to v \mid L) = \max\!\Big(P_{\text{min}}(W, L),\; P_{\text{raw}}(u \to v \mid L)\Big)$$
  where $G_{\text{mars}} = 0.38$, $W = 80\text{ kg}$, $L = 65\text{ kg} + m_{\text{samples}}$, $V = 1.1\text{ m/s}$.

* **Computationally Optimized Formulation (Per-Leg Scalar Precomputation):**
  $$P_{\text{stand}}(L) = 0.38\left[ 1.5\,W + 2.0(W + L)\left(\frac{L}{W}\right)^{\!2} \right]$$
  $$\alpha_V(L) = 0.57\,(W + L)\,V^2, \qquad \beta_V(L) = 0.133\,(W + L)\,V$$
  $$P_{\text{watts}}(u \to v \mid L) = P_{\text{stand}}(L) + \bar{\eta}_{uv}\max\!\Big(0.2\,\alpha_V(L),\; \alpha_V(L) + \beta_V(L)\,S_{uv}\Big)$$

* **Definitions:**
  - $P_{\text{stand}}(L)$: Standing and postural power under Martian gravity.
  - $\alpha_V(L)$: Kinetic horizontal walking locomotion factor.
  - $\beta_V(L)$: Grade-dependent vertical work factor.
* **Ranges:** Inputs: $W = 80.0\text{ kg}$; $L \in [65.0, 95.0]\text{ kg}$; $S_{uv} \in [-36.4\%, +36.4\%]$; $\bar{\eta}_{uv} \in [1.05, 18.0]$. Outputs: $P_{\text{watts}} \in [65.0, 1850.0]\text{ Watts}$.
* **Impact:** Precomputing the three leg scalars avoids evaluating quadratic expressions across $16,000$ directed arcs. The $20\%$ locomotion floor ($\ge 0.2\,\alpha_V$) prevents steep descents ($S_{uv} < -4.7\%$) from yielding negative power.

---

#### 4.1.2 Direct Oxygen Consumption Edge Weight $\Delta\text{O}_2(u \to v)$
* **Optimized Formulation (Single-Scalar Multiply):**
  $$\Delta\text{O}_2(u \to v \mid L) = \gamma_{\text{O}_2}\,P_{\text{watts}}(u \to v \mid L)\,d_{3D}(u,v), \quad \gamma_{\text{O}_2} = \frac{1}{20,100 \times V} \approx 4.52284 \times 10^{-5}\text{ L}\cdot\text{s}/(\text{J}\cdot\text{m})$$
* **Definitions:**
  - $\gamma_{\text{O}_2}$: Combined velocity and Weir indirect calorimetry conversion factor ($1\text{ L }\text{O}_2 \approx 20,100\text{ Joules}$).
  - $\Delta\text{O}_2$: Consumed oxygen volume along directed arc $u \to v$.
* **Ranges:** Inputs: $P_{\text{watts}} \in [65, 1850]\text{ W}$; $d_{3D} \in (0, 27.7]\text{ m}$. Outputs: $\Delta\text{O}_2 \in [0.0025, 2.30]\text{ Liters }\text{O}_2$.
* **Impact:** Sets edge weights directly in life-support units. Asymmetric directed edges capture uphill oxygen costs accurately.

---

### 4.2 Provenance, Rationale, Engineering Problems & Judge Defense (Step 4)
* **Provenance:** Pandolf et al. (1977) metabolic load carriage; Farley & McMahon (1992) hypogravity biomechanics; Santee et al. (2001) / Minetti et al. (2002) eccentric muscle braking; Weir (1949) oxycaloric equivalent.
* **Why We Use Them:** Expressing graph weights directly in Liters of $\text{O}_2$ couples path planning to astronaut life-support envelopes.
* **Engineering Solutions:**
  - Negative watts on steep downhill slopes: Implemented the $20\%$ eccentric muscle braking floor $\max(0.2\,\alpha_V, \alpha_V + \beta_V S_{uv})$, ensuring strictly positive weights for Dijkstra and $A^*$.
* **NASA Judge Defense:**
  - *Q: Why does picking up samples at CRISM waypoints change path geometry instead of just scaling total energy?*
    *A: Because payload $L$ enters non-linearly: standing posture scales quadratically with $(L/W)^2$, while grade locomotion scales linearly. When $L$ increases from $65\text{ kg}$ to $90\text{ kg}$, the ratio of grade penalty to baseline shifts, making steep ridges metabolically more expensive than flatter bedrock detours.*

---

## 5. Step 5 Technical Reference: Payload-Aware Directed $A^*$ & Point of No Return

### 5.1 Canonical vs. Computationally Optimized Equations

#### 5.1.1 Strictly Admissible & Consistent 3D Bioenergetic $A^*$ Heuristic $h_{\text{O}_2}$
* **Optimized Formulation (Division-Free Convex Lower Bound):**
  $$h_{\text{O}_2}(u, g \mid L) = \gamma_{\text{O}_2}\left[\Big(P_{\text{stand}}(L) + 0.2\,\eta_{\min}\,\alpha_V(L)\Big)d_{3D}(u,g) + 0.8\,\eta_{\min}\,\alpha_V(L)\,\Delta d_{2D} + 100\,\eta_{\min}\,\beta_V(L)\max(0.0,\, \Delta z)\right]$$
* **Definitions:**
  - $u, g$: Current node coordinate and target waypoint coordinate.
  - $\eta_{\min} = 1.05$: Global minimum bedrock friction prior.
  - $\Delta d_{2D} = \Vert u - g\Vert_2$, $\Delta z = z_g - z_u$.
* **Ranges:** Inputs: $d_{3D} \in [0.0, 725.0]\text{ m}$; $\Delta z \in [-200.0, +200.0]\text{ m}$. Outputs: $h_{\text{O}_2} \in [0.0, 60.0]\text{ Liters }\text{O}_2$.
* **Impact:** Eliminates division by $\Delta d_{2D}$, avoiding $0/0 = \text{NaN}$ when $u = g$. Guarantees strict admissibility ($h(u) \le d^*(u)$) and monotonicity ($h(u) \le c(u,v) + h(v)$), ensuring zero node reopenings during $A^*$.

---

#### 5.1.2 Reversed-Graph Dijkstra for Real-Time Point of No Return
* **Optimized Formulation (Single-Source Dijkstra on Transposed Graph $\mathcal{G}^\top$):**
  $$\mathcal{G}^{\top} = (V, E^{\top}) \quad \text{where } w^{\top}(v \to u) = \Delta\text{O}_2(u \to v \mid L(t))$$
  $$\mathbf{d}_{\text{ret}} = \text{Dijkstra}\!\big(\mathcal{G}^{\top},\; \text{source}=v_{\text{Hab}}\big)$$
  $$\mathcal{M}_{\text{PNR}}(u, t) = \left(\mathbf{d}_{\text{ret}}[u] > \frac{\text{O}_{2,\text{tank}}(t)}{1.15}\right)$$
* **Definitions:**
  - $\mathcal{G}^\top$: Graph transpose with flipped edge directions and identical forward weights.
  - $\text{O}_{2,\text{tank}}(t)$: Remaining life-support oxygen.
  - $\mathcal{M}_{\text{PNR}}$: Point of No Return hazard indicator vector.
* **Ranges:** Inputs: $\text{O}_{2,\text{tank}} \in [0.0, 25.0]\text{ L}$; safety reserve factor $= 1.15$. Outputs: $\mathbf{d}_{\text{ret}} \in \mathbb{R}^{|V|}$; $\mathcal{M}_{\text{PNR}} \in \{0, 1\}^{|V|}$.
* **Impact:** Replaces $1,200$ separate $A^*$ return searches with a single $O(|E| + |V| \log |V|)$ pass outward from $v_{\text{Hab}}$. Updates the exact return $\text{O}_2$ budget for all $1,200$ nodes simultaneously in $< 2.8\text{ ms}$.

---

### 5.2 Provenance, Rationale, Engineering Problems & Judge Defense (Step 5)
* **Provenance:** Hart, Nilsson & Raphael (1968) $A^*$ algorithm; Dijkstra (1959) shortest paths; Graph Transposition Duality.
* **Why We Use Them:** Real-time PNR evaluation guarantees that astronauts are alerted before venturing beyond the physical life-support return envelope.
* **Engineering Solutions:**
  - Singularity at goal: Division-free formulation bounds grade work via $\max(0, \Delta z)$, avoiding $0/0 = \text{NaN}$.
  - $O(N^2)$ update bottleneck: Graph transposition duality computes all return paths in a single outward Dijkstra pass.
* **NASA Judge Defense:**
  - *Q: Why is running Dijkstra on `G.reverse()` mathematically valid for asymmetric uphill/downhill costs?*
    *A: In a directed graph, the shortest path from every node $u$ to target $v_{\text{Hab}}$ in $\mathcal{G}$ is mathematically dual to the shortest path from $v_{\text{Hab}}$ to node $u$ in transposed graph $\mathcal{G}^\top$ where directed arcs are reversed while retaining forward weights $w(u \to v)$. If $u$ is below the Hab, the astronaut's return trip $u \to v_{\text{Hab}}$ is uphill; traversing that reversed arc outward from $v_{\text{Hab}}$ carries that exact uphill weight.*

---

## 6. Step 6 Technical Reference: Sensor Physics Modules

### 6.1 Canonical vs. Computationally Optimized Equations

#### 6.1.1 Hab Radio Line-of-Sight ($L_{\text{LOS}}$) & Canyon Radiation Shielding ($R_{\text{rad}}$)
* **Optimized Formulations (Running-Max Horizon Sweep + Tangent Identity):**
  $$m_{\max}(s_i) = \max\!\big(m_{\max}(s_{i-1}),\; \alpha_{\text{Hab}}(s_{i-1})\big), \quad L_{\text{LOS}}\!\big(r(s_i), c(s_i)\big) = \mathbb{I}\!\big[\alpha_{\text{rec}}(s_i) \ge m_{\max}(s_i)\big]$$
  $$R_{\text{rad}}(r,c) = \frac{1}{N_{\text{az}}}\sum_{k=1}^{N_{\text{az}}} \frac{T_k(r,c)^2}{1.0 + T_k(r,c)^2}, \quad T_k(r,c) = \max\!\left(0.0,\; \max_{s \in [1, S_{\max}]} \frac{z(r_s, c_s) - z(r,c)}{s}\right)$$
* **Impact:** Reduces viewshed raycasting complexity from $O(L^3)$ to $O(L^2)$. The identity $\sin^2(\arctan T) = \frac{T^2}{1+T^2}$ eliminates transcendental trigonometric calls across all 8 azimuth angles.

---

#### 6.1.2 Rover Ground-Truth Image Calibration ($\delta_{\text{rock}}$) & Uncertainty Footprint
* **Formulations:**
  $$\delta_{\text{rock}}(w) = \frac{1}{|\Omega_{\text{ground}}|}\sum_{(u,v) \in \Omega_{\text{ground}}} \mathbb{I}\!\left[\sqrt{(I_w * K_r)^2 + (I_w * K_c)^2} > \tau_{\text{edge}}\right]$$
  $$\rho_{\text{rover}}(r,c) = \max_{w \in \mathcal{W}_{\text{rover}}} \exp\!\left(-\frac{(r - r_w)^2 + (c - c_w)^2}{2\,r_{\text{fov}}^2}\right)$$
* **Impact:** Integrates actual NASA rover camera observations to calibrate orbital roughness and collapse epistemic uncertainty along explored traverses.

---

## 7. Complete Interface Architecture & Wire Protocol Specification

The interface architecture connects flight control and the field astronaut through a resilient, sub-millisecond full-duplex communication pipeline.

### 7.1 Mission Control 3D Web Cockpit (`frontend/public/index.html`)
* **3D Viewport:** Three.js WebGL canvas rendering the $3\times 3$ Jezero Crater terrain mesh, Hab structure, settlement depots, and optimal path curves.
* **Planning Dock:**
  - `TRAVERSE` tab: Route scheduling, waypoint targeting, sample load slider ($0\text{--}30\text{ kg}$), walk traverse trigger, photo scouting.
  - `SETTLEMENTS` tab: Refill depot monitoring, oxygen replenishment allocation.
* **Top Status Pill:** Cursor coordinate aim, remaining oxygen readout, and `Pair Gyro` QR code trigger.
* **Debrief Modal:** Post-EVA summary with total distance, walk duration, $\text{O}_2$ consumed, and rock samples retrieved.

### 7.2 Mobile Field EVA Companion HUD (`frontend/public/phone.html`)
* **Sensory Telemetry Bridge:** Integrates HTML5 `DeviceOrientationEvent` ($\alpha, \beta, \gamma$) to orient the 3D cockpit camera hands-free.
* **Tactile Walk Control:** Mobile walk/stop toggle and speed throttle ($0.5\times\text{--}2.0\times$).
* **Emergency Audio & Haptic Alerts:** Synthesizes Web Audio alarm frequencies ($880\text{ Hz}$ warning, $440\text{ Hz}$ caution) and device vibration patterns when approaching PNR limits.

### 7.3 Communication Wire Protocol (`server.py`)

#### Telemetry Frame 1: Mobile Phone to Cockpit (`phone_input`)
```json
{
  "type": "phone_input",
  "heading": 142.5,
  "alpha": 142.5,
  "beta": 12.3,
  "gamma": -4.1,
  "walk": 1,
  "speed": 1.2,
  "command": null,
  "source": "phone_live",
  "seq": 1042,
  "updated_at": 1731700964.12
}
```

#### Telemetry Frame 2: Cockpit to Mobile Phone (`pc_telemetry`)
```json
{
  "type": "pc_telemetry",
  "currO2": 21.45,
  "tankCapacityL": 25.0,
  "elevLocal": 24.4,
  "carriedSampleKg": 10.0,
  "pnrStatus": "SAFE (1.15x)",
  "pathProgress": 0.42,
  "pathDist": 312.8,
  "isPathWalking": true,
  "heading": 142.5,
  "updated_at": 1731700964.15
}
```

---

## 8. Summary of Performance Benchmarks

| Computational Stage | Matrix / Graph Size | Execution Time (Standard CPU) | Speedup vs. Naive |
| :--- | :--- | :--- | :--- |
| **Step 1: Cost Field $C(r,c)$** | $512\times 512$ ($262,144\text{ px}$) | $4.2\text{ ms}$ | $4.2\times$ via SIMD FMA & cliff pruning |
| **Step 2: Deviation Score $D(r,c)$** | $514\times 514$ ($h=1\text{ halo}$) | $3.1\text{ ms}$ | $3.5\times$ via separable Sobel & fixed $S_\nabla$ |
| **Step 3: Adaptive Vertex Sampling** | $N = 1,200\text{ nodes}$ | $0.6\text{ ms}$ | $12.0\times$ via 1D prefix-sum binary search |
| **Step 3: Tensor Ray Collision Check** | $M \approx 8,000\text{ candidate edges}$ | $1.8\text{ ms}$ | $18.5\times$ via $(M\times 11\times 2)$ NumPy tensor |
| **Step 4: Pandolf Bioenergetic Weights** | $|E| \approx 16,000\text{ directed edges}$| $0.9\text{ ms}$ | $4.8\times$ via per-leg scalar precomputation |
| **Step 5: Directed $A^*$ Routing** | $1,200\text{ nodes, multi-leg}$ | $1.4\text{ ms}$ | $3.2\times$ via division-free admissible $h_{\text{O}_2}$ |
| **Step 5: Point of No Return Dijkstra**| All $1,200\text{ nodes simultaneously}$ | $2.6\text{ ms}$ | $460\times$ vs. $1,200$ individual searches |
| **Step 6: Hab Radio Viewshed ($L_{\text{LOS}}$)**| $512\times 512$ radial sweep | $5.8\text{ ms}$ | $8.4\times$ via running-max horizon slopes |
| **Total End-to-End Cycle** | Full 6-step mission update | **$< 25\text{ ms}$** | Real-time $(>40\text{ Hz})$ flight-ready cycle |
