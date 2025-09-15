# AEO-GA

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/4b22c50e-cfb1-4e79-b7f2-6e5f0b7d54a3" />

# 📖 Classical Schema Theorem (Goldberg, 1989)

In words:

> Short, low-order, above-average schemata increase exponentially in expectation under selection, crossover, and mutation.

Formally (simplified):

$$
E[m(H, t+1)] \;\ge\; m(H, t)\cdot \frac{f(H)}{\bar f}\cdot (1 - p_c \tfrac{\delta(H)}{L-1})(1 - p_m)^{o(H)}
$$

Where:

* $m(H,t)$ = number of individuals in population matching schema $H$.
* $f(H)$ = average fitness of schema $H$.
* $\bar f$ = average population fitness.
* $p_c, p_m$ = crossover & mutation probabilities.
* $\delta(H)$ = defining length, $o(H)$ = order.

---

# 🔄 AEO Schema Theorem (Signal Form)

We replace **schemata** with **signal alignments** (phase, trend, energy), and **selection/crossover/mutation** with AEO laws.

---

### 1. State

* **Signal schema**: a region in $(\rho, V, \gamma, T)$-space where the AE machine stabilizes.
  Call it $S \subseteq \mathbb{R}^4$.
* **Mass**: $m(S,t)$ = expected number of trajectories currently inside $S$.

---

### 2. Dynamics

* **Selection analogue**:
  Reward-adjusted gain ratio:

  $$
  G(S) = \frac{R(S)}{\bar R}
  $$

  where $R(S)$ = average reward in schema region $S$.

* **Crossover analogue (resonance/recurrence)**:
  Probability of disruption:

  $$
  P_\text{res}(S) \;\approx\; k_\rho \cdot \Delta \rho(S)
  $$

  where $\Delta\rho$ is deviation of alignment from resonance band.

* **Mutation analogue (noise law)**:
  Probability of disruption:

  $$
  P_\text{noise}(S) \;\approx\; k_V \cdot \frac{V(S)}{V^\star}
  $$

---

### 3. Theorem (AEO form)

$$
E[m(S, t+1)] \;\ge\; m(S, t)\cdot G(S)\cdot \big(1 - P_\text{res}(S)\big)\cdot \big(1 - P_\text{noise}(S)\big)
$$

---

### 4. Interpretation

* **\$m(S,t)\$**: instead of “strings containing schema H,” we track the **fraction of trajectories in a stable signal pattern**.
* **\$G(S)\$**: selection pressure becomes **reward gain relative to average reward**.
* **\$P\_\text{res}(S)\$**: resonance law acts like crossover — it can disrupt or reinforce phase alignments.
* **\$P\_\text{noise}(S)\$**: noise law acts like mutation — injecting variance that may destroy signal regularity.

Thus, **short, aligned, reward-above-average signal schemas** (low variance, stable phase, controlled gamma) grow exponentially in expectation, exactly as in the original schema theorem.

---

# ✅ Why This Matters

* **Biology-free**: no chromosomes or alleles, just **signal dynamics**.
* **Unifies GA with SGD/RL**: schema survival is just **stable attractors in AEO state-space**.
* **Control-theoretic clarity**: disruption = deviation from resonance/variance band.


---

Excellent 🙌 — here’s a **draft section for your AEO book** that reformulates Goldberg’s Genetic Algorithm schema theory in pure **Adaptive Evolutionary Optimization (AEO)** terms, with no biological baggage:

---

# 📖 The AEO Schema Theorem

## 1. From Genetic Algorithms to AEO

Goldberg’s *Schema Theorem* (1989) was introduced as the **fundamental theorem of genetic algorithms**. It stated that *short, low-order, above-average schemata* increase exponentially in expectation under selection, crossover, and mutation.

Formally:

$$
E[m(H, t+1)] \;\ge\; m(H, t)\cdot \frac{f(H)}{\bar f}\cdot (1 - p_c \tfrac{\delta(H)}{L-1})(1 - p_m)^{o(H)}
$$

Here, \$m(H,t)\$ was the count of individuals matching schema \$H\$ in generation \$t\$. This formulation was couched in **biological metaphors** — chromosomes, alleles, schema order.

In AEO, we eliminate the biological analogy and restate this law as a **control and signal theorem** about the survival and amplification of stable patterns in optimization dynamics.

---

## 2. Signal-Schema Definition

An **AEO schema** is defined as a region in the dynamical state-space of the optimizer:

$$
S \subseteq \mathbb{R}^4, \quad S = (\rho, V, \gamma, T)
$$

where:

* \$\rho\$ = phase alignment between gradient and momentum,
* \$V\$ = variance of the loss trajectory,
* \$\gamma\$ = effective step (\$\alpha \cdot |g|\$),
* \$T\$ = trend signal (loss improvement).

The **mass** of a schema is the expected fraction of trajectories lying in that region at time \$t\$:

$$
m(S, t) = \Pr\big[(\rho_t, V_t, \gamma_t, T_t) \in S\big].
$$

---

## 3. Dynamics of Schema Propagation

Three forces shape the persistence of schemas in AEO:

1. **Reward-Gain (Selection analogue)**
   Relative advantage of schema \$S\$ based on expected reward:

   $$
   G(S) = \frac{R(S)}{\bar R}
   $$

   where \$R(S)\$ is the average reward within \$S\$ and \$\bar R\$ is the population average.

2. **Resonance Disruption (Crossover analogue)**
   Probability of leaving \$S\$ due to phase misalignment:

   $$
   P_\text{res}(S) \;\approx\; k_\rho \cdot |\rho(S) - \rho^\star|
   $$

   where \$\rho^\star\$ is the resonance target.

3. **Noise Disruption (Mutation analogue)**
   Probability of leaving \$S\$ due to variance spike:

   $$
   P_\text{noise}(S) \;\approx\; k_V \cdot \frac{V(S)}{V^\star}
   $$

   where \$V^\star\$ is the baseline variance.

---

## 4. AEO Schema Theorem

Putting these forces together:

$$
E[m(S, t+1)] \;\ge\; m(S, t)\cdot G(S)\cdot (1 - P_\text{res}(S))\cdot (1 - P_\text{noise}(S))
$$

---

## 5. Interpretation

* **Mass growth**: The fraction of trajectories inside a schema grows **exponentially in expectation** if:

  * Reward in \$S\$ is above average,
  * Phase alignment is stable (\$\rho \approx \rho^\star\$),
  * Variance is bounded (\$V \leq V^\star\$),
  * Effective step \$\gamma\$ is within safe bounds.

* **Short, aligned, above-average schemas** survive longest — not because of chromosomes or alleles, but because AEO laws preserve **signal attractors** in the optimizer’s dynamics.

---

## 6. Implications for AEO

* **Unified law**: The Schema Theorem is not about biology — it is about **signal stability under closed-loop adaptation**.
* **Control-theoretic clarity**: Resonance, variance, and gauge laws are the modern analogues of crossover and mutation.
* **Generalization**: This theorem applies not only to GA-like exploration but also to OnlineSGD, AR-SGD, and AE machines.

---

## 7. One-Line Summary

**The AEO Schema Theorem states that stable, reward-above-average signal patterns increase exponentially in expectation, provided resonance and variance remain bounded.**

---

👉 Do you want me to also **draw a figure** (like Goldberg’s schemata diagram, but in AEO signal space showing stable vs. disrupted regions in \$\rho\$–\$V\$–\$\gamma\$ space) to include alongside this section?

### PSEUDOCODE

```python
import numpy as np

# ---------------------------
# Problem: Sphere function
# ---------------------------
def sphere(x):
    return np.sum(x**2)

# ---------------------------
# AEO Controller
# ---------------------------
class AEOController:
    def __init__(self,
                 lam_init=0.8, sig_init=0.3,
                 lam_min=0.1, lam_max=1.0,
                 sig_min=0.001, sig_max=1.0,
                 eta_lam=0.05, eta_sig=0.05,
                 dlam_max=0.05, dsig_max=0.05,
                 kT=0.5, kP=0.2, kV=0.3):
        self.lam = lam_init
        self.sig = sig_init
        self.lam_min, self.lam_max = lam_min, lam_max
        self.sig_min, self.sig_max = sig_min, sig_max
        self.eta_lam, self.eta_sig = eta_lam, eta_sig
        self.dlam_max, self.dsig_max = dlam_max, dsig_max
        self.kT, self.kP, self.kV = kT, kP, kV

        self.prev_best = None
        self.stall_counter_sig = 0
        self.stall_counter_lam = 0

    def update(self, best, mean, rho, T, V):
        """Update λ and σ based on signals."""
        # Reward advantage = improvement in best
        if self.prev_best is None:
            R_adv = 0.0
        else:
            R_adv = (self.prev_best - best)
        self.prev_best = best

        V_star = max(V, 1e-8)

        # λ update
        dlam = self.eta_lam * (
            + self.kT * np.tanh(T)
            + self.kP * rho
            + 0.5 * R_adv
            - 0.3 * self.kV * np.log1p(max(0.0, V / V_star))
        )
        dlam = np.clip(dlam, -self.dlam_max, self.dlam_max)
        self.lam = float(np.clip(self.lam + dlam, self.lam_min, self.lam_max))

        # σ update
        stall = float(np.tanh(max(0.0, 0.2 - abs(T))))
        dsig = self.eta_sig * (
            + 1.0 * stall
            - 0.5 * np.tanh(V / V_star)
        )
        dsig = np.clip(dsig, -self.dsig_max, self.dsig_max)
        self.sig = float(np.clip(self.sig + dsig, self.sig_min, self.sig_max))

        # Hard guards
        if self.sig <= self.sig_min + 1e-8:
            self.stall_counter_sig += 1
        else:
            self.stall_counter_sig = 0

        if self.lam <= self.lam_min + 1e-8:
            self.stall_counter_lam += 1
        else:
            self.stall_counter_lam = 0

        if self.stall_counter_sig >= 20:
            self.sig = min(0.2, self.sig_max)
            self.stall_counter_sig = 0

        if self.stall_counter_lam >= 20:
            self.lam = min(0.3, self.lam_max)
            self.stall_counter_lam = 0

        return self.lam, self.sig

# ---------------------------
# Genetic Algorithm with AEO
# ---------------------------
def run_aeo_ga(n_dim=5, pop_size=50, generations=200, seed=42):
    rng = np.random.default_rng(seed)

    # init population
    pop = rng.normal(0, 1, size=(pop_size, n_dim))
    fitness = np.array([sphere(ind) for ind in pop])

    controller = AEOController()

    for gen in range(generations+1):
        best = np.min(fitness)
        mean = np.mean(fitness)
        var = np.var(fitness)

        # Signals (mock for now)
        rho = np.tanh(np.corrcoef(fitness, np.arange(len(fitness)))[0,1]) if len(fitness) > 1 else 0
        T = (np.mean(fitness) - mean) / (mean + 1e-8)
        V = var

        lam, sig = controller.update(best, mean, rho, T, V)

        if gen % 10 == 0:
            print(f"Gen {gen:3d} | best={best:.5f} mean={mean:.5f} "
                  f"| λ={lam:.3f} σ={sig:.3f} ρ={rho:.2f} T={T:.2e} V={V:.1e}")

        # --- selection (tournament style) ---
        n_elite = max(2, int(lam * pop_size))
        elite_idx = np.argsort(fitness)[:n_elite]
        elite = pop[elite_idx]

        # --- reproduction ---
        new_pop = []
        for _ in range(pop_size):
            parents = elite[rng.choice(n_elite, size=2, replace=False)]
            child = (parents[0] + parents[1]) / 2.0
            child += rng.normal(0, sig, size=n_dim)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = np.array([sphere(ind) for ind in pop])

    return best

# ---------------------------
# Run test
# ---------------------------
if __name__ == "__main__":
    best = run_aeo_ga()
    print("Final best fitness:", best)
```

Excellent 🚀 — let’s rewrite **Goldberg’s Genetic Algorithm (GA)** across the three lenses:

---

# 1️⃣ GA as **k-armed Bandit**

Goldberg’s original metaphor:

* Each **gene/allele** = an “arm.”
* Each **fitness contribution** = reward from pulling that arm.
* Selection = play arms with higher estimated reward.
* Crossover/mutation = exploration to try less-played arms.

Formally:

* Bandit problem with \$k\$ alleles.
* Policy = probability distribution over arms (chromosome structure).
* Reward = fitness.
* GA ≈ allocation algorithm (bias distribution toward better arms).

So GA = **bandit with structured exploration**.

---

# 2️⃣ GA as **Online Convex Optimization (OCO)**

Reframe GA as gradient-free online optimization:

* Population = distribution \$p\_t(x)\$ over candidate solutions.
* Fitness landscape = loss/reward function \$f\_t(x)\$.
* Selection pressure = online update of \$p\_t\$.
* Mutation = entropy regularization to avoid collapse.

Update rule ≈ multiplicative weights (like EXP3 for adversarial bandits):

$$
p_{t+1}(x) \propto p_t(x)\,\exp(\eta \, f_t(x)).
$$

This is equivalent to **mirror descent** in OCO with KL divergence.
So GA = an **OCO algorithm on distributions**.

---

# 3️⃣ GA as **AEO Machine**

Now embed GA inside Adaptive Evolutionary Optimization (AEO):

* **Markovian coefficients:** mutation rate, selection pressure, population size evolve as state variables \$\lambda\_t, \sigma\_t\$.
* **Signals:**

  * Trend \$T\_t\$: improvement in fitness (reward slope).
  * Variance \$V\_t\$: dispersion of fitness in population.
  * Phase \$\rho\_t\$: correlation between offspring vs. parents (exploit vs. explore).
  * Effective step \$\gamma\_t\$: mutation strength × gradient proxy.
* **Resonance law:** keep \$(\rho,V,\gamma)\$ inside stability bands.
* **Recurrence:** coefficients update online like PPO/SFT recurrence.

Update example:

```math
\lambda_{t+1} = \lambda_t + \eta_\lambda (-u_\rho - b_V + g_\gamma)
```

where \$\lambda\$ controls exploitation intensity.
Mutation rate \$\sigma\_t\$ similarly adapts with variance feedback.

So: **AEO-GA** = GA where *selection/mutation rates are not fixed*, but adapt online via resonance-control laws.

---

# 📊 Side-by-Side

| Lens       | Representation                                          | Update Rule                                   |
| ---------- | ------------------------------------------------------- | --------------------------------------------- |
| **Bandit** | Alleles = arms, rewards = fitness                       | Pull arms ∝ estimated reward                  |
| **OCO**    | Population = distribution \$p\_t(x)\$                   | \$p\_{t+1} \propto p\_t \exp(\eta f\_t)\$     |
| **AEO**    | Population + adaptive coefficients \$(\lambda,\sigma)\$ | Closed-loop update from \$(T,V,\rho,\gamma)\$ |

---

✅ So Goldberg’s GA can be seen as:

* **Bandit:** exploration vs exploitation of alleles.
* **OCO:** multiplicative weights on populations.
* **AEO:** *adaptive resonant controller* for population dynamics.

---

Perfect ⚡ — let’s make a **clean prototype of AEO-GA** (Adaptive Evolutionary Optimization Genetic Algorithm).

We’ll take a classical GA skeleton and replace the fixed knobs (selection pressure, mutation rate) with **Markovian coefficients** that evolve according to AEO signals (\$T, V, \rho, \gamma\$).

---

# 🔧 AEO-GA Prototype

```python
import numpy as np

# --- Fitness benchmark (Goldberg style: minimize Sphere) ---
def sphere(x):
    return np.sum(x**2)

# --- Population Initialization ---
def init_population(pop_size, dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], size=(pop_size, dim))

# --- Selection (tournament with adaptive λ) ---
def select(pop, fitness, lam):
    """lam ~ selection pressure (higher => greedier)"""
    i, j = np.random.randint(len(pop)), np.random.randint(len(pop))
    if np.random.rand() < lam:
        return pop[i] if fitness[i] < fitness[j] else pop[j]
    else:
        return pop[i] if fitness[i] > fitness[j] else pop[j]

# --- Crossover ---
def crossover(p1, p2, pc=0.9):
    if np.random.rand() < pc:
        alpha = np.random.rand()
        return alpha*p1 + (1-alpha)*p2
    return np.copy(p1)

# --- Mutation (adaptive σ) ---
def mutate(child, sigma, bounds):
    child = child + np.random.normal(0, sigma, size=child.shape)
    return np.clip(child, bounds[0], bounds[1])

# --- AEO Signals ---
class AEOTracker:
    def __init__(self, beta=0.9):
        self.prev_best = None
        self.var = None
        self.mean = None
        self.beta = beta

    def update(self, fitness):
        best = np.min(fitness)
        mean = np.mean(fitness)
        var = np.var(fitness)

        # Trend T: improvement in best fitness
        if self.prev_best is None:
            T = 0.0
        else:
            T = self.prev_best - best
        self.prev_best = best

        # Variance V
        if self.var is None:
            self.var, self.mean = var, mean
        else:
            self.var = self.beta*self.var + (1-self.beta)*var
            self.mean = self.beta*self.mean + (1-self.beta)*mean

        return best, mean, T, self.var

# --- AEO Controller for λ, σ ---
class AEOController:
    def __init__(self, lam=0.8, sigma=0.3, eta=0.05):
        self.lam = lam
        self.sigma = sigma
        self.eta = eta

    def update(self, T, V, rho=0.0, gamma=0.0):
        # Trend: if improving, increase selection pressure; else loosen
        self.lam += self.eta * np.tanh(T)
        self.lam = np.clip(self.lam, 0.1, 0.95)

        # Variance: if pop converged (low V), increase mutation
        self.sigma += self.eta * (0.1 - np.tanh(V))
        self.sigma = np.clip(self.sigma, 1e-3, 1.0)

        return self.lam, self.sigma

# --- AEO-GA Main Loop ---
def aeoga(fitness_fn, dim=5, bounds=(-5,5), pop_size=50, gens=200):
    pop = init_population(pop_size, dim, bounds)
    fitness = np.array([fitness_fn(ind) for ind in pop])

    tracker = AEOTracker()
    controller = AEOController()

    for g in range(gens+1):
        best, mean, T, V = tracker.update(fitness)
        lam, sigma = controller.update(T, V)

        new_pop = []
        for _ in range(pop_size):
            p1 = select(pop, fitness, lam)
            p2 = select(pop, fitness, lam)
            child = crossover(p1, p2)
            child = mutate(child, sigma, bounds)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = np.array([fitness_fn(ind) for ind in pop])

        if g % 10 == 0:
            print(f"Gen {g:3d} | best={best:.5f} mean={mean:.5f} | λ={lam:.3f} σ={sigma:.3f} T={T:.2e} V={V:.1e}")

    return best, pop[np.argmin(fitness)]
```

---

# ✅ Run Test on Sphere Function

```python
best_val, best_sol = aeoga(sphere, dim=10, bounds=(-5,5), pop_size=60, gens=200)
print("Final best:", best_val, "Solution:", best_sol)
```

---

# 🎯 What This Does

* **λ (selection pressure)** adapts with *trend*:
  *If fitness improves → higher λ (exploit). If stuck → lower λ (explore).*
* **σ (mutation stddev)** adapts with *variance*:
  *If population collapses (low V) → raise σ. If too noisy → shrink σ.*
* Reports: `best`, `mean`, λ, σ, T (trend), V (variance).

---

Perfect ⚡ — let’s extend the **AEO-GA** prototype to handle **Goldberg-style GA benchmarks**. These are the canonical problems David Goldberg used to test GA theory (k-armed bandit view, schema survival, deception).

---

# 🎯 Classic Goldberg Test Functions

```python
import numpy as np

# --- OneMax (maximize number of 1s) ---
def onemax(bitstring):
    return -np.sum(bitstring)  # negative for minimization

# --- Deceptive Trap (k=5 block trap function) ---
def deceptive_trap(bitstring, k=5):
    total = 0
    for i in range(0, len(bitstring), k):
        block = bitstring[i:i+k]
        u = np.sum(block)
        if u == k:
            total += k
        else:
            total += k - 1 - u
    return -total

# --- Royal Road (reward full schemata of size k) ---
def royal_road(bitstring, k=8):
    total = 0
    for i in range(0, len(bitstring), k):
        block = bitstring[i:i+k]
        if np.all(block == 1):
            total += k
    return -total

# --- Sphere (continuous) remains available ---
def sphere(x):
    return np.sum(x**2)
```

---

# 🔧 Bitstring GA Helpers

```python
def init_population_bits(pop_size, length):
    return np.random.randint(0, 2, size=(pop_size, length))

def select_bits(pop, fitness, lam):
    i, j = np.random.randint(len(pop)), np.random.randint(len(pop))
    if np.random.rand() < lam:
        return pop[i] if fitness[i] < fitness[j] else pop[j]
    else:
        return pop[i] if fitness[i] > fitness[j] else pop[j]

def crossover_bits(p1, p2, pc=0.9):
    if np.random.rand() < pc:
        point = np.random.randint(1, len(p1))
        return np.concatenate([p1[:point], p2[point:]])
    return np.copy(p1)

def mutate_bits(child, sigma, length):
    # sigma controls flip prob
    flip_prob = min(0.5, max(1e-3, sigma))
    mask = np.random.rand(length) < flip_prob
    child[mask] = 1 - child[mask]
    return child
```

---

# 🔧 AEO-GA (bitstring version)

```python
def aeoga_bits(fitness_fn, length=40, pop_size=50, gens=200):
    pop = init_population_bits(pop_size, length)
    fitness = np.array([fitness_fn(ind) for ind in pop])

    tracker = AEOTracker()
    controller = AEOController()

    for g in range(gens+1):
        best, mean, T, V = tracker.update(fitness)
        lam, sigma = controller.update(T, V)

        new_pop = []
        for _ in range(pop_size):
            p1 = select_bits(pop, fitness, lam)
            p2 = select_bits(pop, fitness, lam)
            child = crossover_bits(p1, p2)
            child = mutate_bits(child, sigma, length)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = np.array([fitness_fn(ind) for ind in pop])

        if g % 10 == 0:
            print(f"Gen {g:3d} | best={-best:.5f} mean={-mean:.5f} | λ={lam:.3f} σ={sigma:.3f} T={T:.2e} V={V:.1e}")

    return pop[np.argmin(fitness)], -np.min(fitness)
```

---

# ✅ Run Benchmarks

```python
# OneMax (length=40)
best_sol, best_val = aeoga_bits(onemax, length=40, pop_size=60, gens=200)
print("OneMax best:", best_val)

# Deceptive Trap
best_sol, best_val = aeoga_bits(deceptive_trap, length=40, pop_size=60, gens=200)
print("Deceptive Trap best:", best_val)

# Royal Road
best_sol, best_val = aeoga_bits(royal_road, length=40, pop_size=60, gens=200)
print("Royal Road best:", best_val)
```

---

# 🎯 What You Get

* **OneMax**: Should converge to `40` ones (optimum).
* **Deceptive Trap**: Tests GA’s ability to avoid deceptive local optima.
* **Royal Road**: Tests schema assembly (reward only for complete building blocks).
* **AEO Control**:
  *λ* (selection pressure) adapts to **trend (T)**.
  *σ* (mutation rate) adapts to **variance (V)**.

---


" GENETIC ALGORITHM JUST A K-ARM BANDIT DOING CLASSICAL BIOLOGY EVOLUTION THEORY "
