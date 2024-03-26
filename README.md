# Compte Rendu: Heston Model and Sampling from $\chi^2$ Law

## The Heston Model

The Heston model is a stochastic volatility model due to Heston ('92), who modeled the volatility as Cox, Ingersoll and Ross did for modeling interest rates ('85).

### Definition

$$
\begin{align*}
dS_t &= rS_t dt + \sqrt{V_t} S_t dW_t \\
dV_t &= -k(V_t - V^0) dt + \sigma \sqrt{V_t} dZ_t
\end{align*}
$$

- $V_t$ is the instantaneous variance
- $\sigma$ is called the "volatility of volatility"
- $W_t$, $Z_t$ are two Brownian motions correlated with correlation $\rho$

**Remark:** For the implementation, we used the linear discretisation:

$$
\begin{align*}
S_{t+\Delta t} &= S_t + r S_t \Delta t + \sqrt{v_t}S_T\sqrt{\Delta t}(\rho G_1 + \sqrt{1-\rho^2} G_2) \\
v_{t+\Delta t} &= k (v_0 - v_t)\Delta t + \sigma \sqrt{f(v_t)}\sqrt{\Delta t } G_1
\end{align*}
$$

## I : Monte Carlo Heston Pricing

Using the provided Monte Carlo Black Scholes pricing code, we adapted it to simulate the Heston model dynamics by replacing the square root in the volatility process by 0 if it's negative.

| Number of scenarios | CPU execution time | GPU execution time |
|---------------------|---------------------|---------------------|
| 1000                | 0.75s               | 0.176 ms            |
| 100000              | 8.79s               | 0.514 ms            |
| 1000000             | 82.962s             | 4.196 ms            |

More experiments can be conducted using the notebook: [Question1.ipynb](https://colab.research.google.com/drive/1BAh7cddPIS6bEqr8BZ8PM1d6eNOcg2Lr?usp=sharing) and the CUDA file [MC.cu](https://drive.google.com/file/d/1TKDpSfunHFTHZsbkzml0BKSgCnMMIWQt/view?usp=sharing).

## II : Sampling from Decentred $\chi^2$

### Methodology

1. Defined multiple device functions: `GS_star`, `GKM1`, `GKM2`, `GKM3` which perform gamma simulation, and then we simulate using `generate_gamma` and `chi2` as described in the paper.
2. We use the kernel function that:
    - Initializes a random number generator (curandState).
    - Generates a Poisson random variable to add the 'non-centrality' aspect to the chi-squared distribution.
    - Runs device functions as intended on the threads using the curand states in a way that ensures statistical independence.

### Performance Comparison

#### Performance on GPU (Nvidia T4)

| Number of samples | Time taken |
|-------------------|------------|
| 10000             | 0.005807s  |
| 1000000           | 0.048577s  |
| 10000000          | 0.341024s  |
| 100000000         | 2.452472s  |

#### Performance on CPU (AMD Ryzen 9 5900HS)

| Number of samples | Time taken |
|-------------------|------------|
| 10000             | 0.512911s  |
| 1000000           | 51.430150s |

More experiments can be conducted using the notebook: [Question2.ipynb](https://colab.research.google.com/drive/17tyFrsFcBfQihJZTCqQy8lvlWCsX4aqD?usp=sharing) and the CUDA file [chi2.cu](https://drive.google.com/file/d/10GqIdHXKK4jsgM0kjzpTTChnsbuH2uuW/view?usp=sharing).

## III : The Heston Model Without Euler

### Methodology: 3 Steps

1. We compute the variance using the conditional law:
    - We fix $V_u$ and we use Chi2 from the previous question to compute $V_t$ given $V_u$ with the parameters given in the article.
2. We simulate the integral of variance using the simulated variance:
    - We start by coding the characteristic function of the integral given $V_t$ and $V_u$.
    - We use this characteristic function to compute the cumulative distribution function. In fact, this leads to an infinite sum. However, we select an integer $N$ at which the summation can be terminated. The higher $N$ is, the more accurate our computation becomes.
    - Then we use the (Inverse) Distribution Function Method for sampling. And since we don't have access to a closed form of $F^{-1}$, we use Newton's method to compute $x$ such that $F(x) = U$.
3. We generate the next instance of the asset price using the computed elements and a formula provided in the article.

Link for our code for this question: [Code for Question 3](https://colab.research.google.com/drive/1POUrM9oMvD_cq54c3iEmS0WMcTfAmBa8?usp=sharing).
