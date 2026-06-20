*A short note on motivation: I wrote this up as a self-contained proof that unifies harmonic analysis and theoretical deep learning. The universal approximation theorem usually appears as a functional-analytic black box (Hahn–Banach, annihilating measures, contradiction); I wanted to see if you could derive the same result from a more concrete handle — the Fejér theorem — and end up with an argument that's easier to teach.*

<div class="abstract">
<span class="env-label">Abstract</span>
We present a self-contained proof of the universal approximation theorem for single-hidden-layer neural networks with continuous, nonpolynomial activation functions. Our approach decomposes the problem into two density arguments: first, that trigonometric polynomials (spanned by sines and cosines) are dense in the space of continuous functions via multidimensional Fejér means; and second, that the span of a nonpolynomial activation is dense in the trigonometric polynomials via the Stone–Weierstrass theorem. Composing these two steps yields the classical result that $\mathcal{M}(\sigma)$ is dense in $C(\RR^n)$ for any continuous nonpolynomial $\sigma$, offering a streamlined alternative to the original functional-analytic proofs.
</div>

## Introduction

The universal approximation theorem is a foundational result in the theory of neural networks. It guarantees that single-hidden-layer feedforward networks with a suitable activation function can approximate any continuous function on compact subsets of $\RR^n$ to arbitrary precision. This result was first established by Cybenko [\[1\]](#cybenko1989) for sigmoidal activations, and later extended by Hornik [\[2\]](#hornik1991) and others to broader classes of activation functions.

Most classical proofs rely on the Hahn–Banach theorem and properties of signed measures: one assumes the network space is not dense, derives a nontrivial annihilating measure, and obtains a contradiction.

In this note, we propose an approach that combines elements from both proofs, while ultimately yielding an elementary argument. We divide the proof into two transparent steps:

1. **Trigonometric density (Lemma 1):** The space $T_n = \text{span}\{\cos(w\cdot x),\, \sin(w\cdot x) : w\in\RR^n\}$ is dense in $C(\RR^n)$. This follows from the multidimensional Fejér theorem, which provides explicit approximants via Cesàro averages of Fourier partial sums.
2. **Activation density (Lemma 2):** For any continuous nonpolynomial $\sigma$, the span of univariate translates $\{\sigma(\lambda x + \theta)\}$ is dense in $C(\RR)$. This is shown by differentiating to recover all monomials in the closure, then invoking Stone–Weierstrass.

Combining these two facts yields the theorem: any $f\in C(\RR^n)$ is first approximated by trigonometric polynomials, and each trigonometric component is then approximated by elements of $\mathcal{M}(\sigma)$.

## Setup

In this section we introduce the setup for understanding neural networks, density concepts over the space of functions, and their relation to the main result we are trying to prove.

<div class="definition">
<span class="env-label">Definition 1</span>
A <strong>1-layer neural network</strong> with activation function $\sigma:\RR\to\RR$, input dimension $n$, and $m$ hidden neurons is a function $f:\RR^n\to\RR$ of the form:
$$f(x) = \sum_{i=1}^{m} c_i \, \sigma(w_i \cdot x - \theta_i),$$
where $w_i \in \RR^n$ are the weight vectors, $\theta_i \in \RR$ are the thresholds (biases), and $c_i \in \RR$ are the output coefficients. The output is taken to be $1$-dimensional.
</div>

Intuitively, each hidden neuron computes a weighted linear combination of the inputs, shifts it by a bias, and passes the result through a fixed nonlinear activation $\sigma$. The output is then a linear combination of these nonlinear features. The universal approximation theorem asserts that, provided $\sigma$ is continuous and nonpolynomial, such networks can approximate any continuous function arbitrarily well on compact sets, given sufficiently many hidden neurons.

<div class="definition">
<span class="env-label">Definition 2</span>
The space of 1-layer-NN functions is given by:
$$\mathcal{M}(\sigma) = \text{span}\bigl(\sigma(w\cdot x - \theta) \;:\; \theta \in \RR,\; w\in \RR^n\bigr).$$
Equivalently:
$$\mathcal{M}(\sigma) = \left\{\sum_{i=1}^r c_i \,\sigma (w_i \cdot x - \theta_i) \;:\; r \in \NN,\; c_i\in \RR,\; \theta_i \in \RR,\; w_i\in \RR^n\right\}.$$
</div>

**Notation.** Throughout, $C(\RR^n)$ denotes the space of continuous functions $f:\RR^n \to \RR$, i.e. functions of $n$ input variables with $1$-dimensional (real-valued) output.

<div class="definition">
<span class="env-label">Definition 3</span>
Given $S\subset C(\RR^n)$, we say $S$ is <strong>dense</strong> in $C(\RR^n)$ iff: for any compact $K\subset \RR^n$, any $f\in C(\RR^n)$, and any $\epsilon >0$, there exists $g\in S$ such that
$$\sup_{x\in K} |f(x)-g(x)| < \epsilon.$$
</div>

The key goal is to show the following result.

<div class="theorem">
<span class="env-label">Theorem 1</span>
Let $\sigma \in C(\RR)$ be a continuous nonpolynomial function. Then $\mathcal{M}(\sigma)$ is dense in $C(\RR^n)$.
</div>

Theorem 1 is the central result of this paper. It states that any continuous function on $\RR^n$ can be uniformly approximated on compact sets by finite sums of the form $\sum c_i \sigma(w_i \cdot x - \theta_i)$ — that is, by a single-hidden-layer neural network — provided $\sigma$ is continuous and not a polynomial. The strength of this result lies in its generality: the activation function $\sigma$ need not be sigmoidal, bounded, or monotone; it only needs to be continuous and nonpolynomial. In other words, the expressive power of a one-layer network is not an artifact of a particular activation shape, but rather a consequence of the nonlinearity itself.

<div class="definition">
<span class="env-label">Definition 4</span>
Denote
$$T_n = \text{span}\{\cos(w\cdot x),\, \sin(w\cdot x) \;:\; w\in \RR^n\} \subset C(\RR^n).$$
</div>

<div class="lemma">
<span class="env-label">Lemma 1</span>
$T_n$ is dense in $C(\RR^n)$.
</div>

<div class="lemma">
<span class="env-label">Lemma 2</span>
Let $\sigma$ be a fixed continuous nonlinearity. Then $N_{\sigma} := \text{span}\{\sigma(\lambda x + \theta) : \lambda,\theta \in \RR\}$ is dense in $C(\RR)$.
</div>

The proof strategy is now clear. We know by Lemma 1 that the trigonometric polynomials $T_n$ are dense in $C(\RR^n)$, so it suffices to show that $\mathcal{M}(\sigma)$ is dense in $T_n$. Each element of $T_n$ is a finite linear combination of functions of the form $\cos(w\cdot x)$ and $\sin(w\cdot x)$, each of which depends on the input $x$ only through the one-dimensional projection $w\cdot x$. Lemma 2 tells us that univariate translates of $\sigma$ can approximate any continuous function of a single variable — in particular, $\cos(\cdot)$ and $\sin(\cdot)$ — on compact intervals. By composing these univariate approximations with the linear projections $x\mapsto w\cdot x$, we obtain elements of $\mathcal{M}(\sigma)$ that approximate each trigonometric component, and hence any element of $T_n$.

<div class="proof">
<span class="env-label">Proof of Theorem 1 (using Lemmas 1 and 2)</span>
Let $f\in C(\RR^n)$, $K\subset \RR^n$ compact, and $\epsilon > 0$. By density of $T_n$ in $C(\RR^n)$, there exists
$$g(x) = \sum_{i=1}^r c_i \, g_i(w_i\cdot x) \in T_n$$
such that $\sup_{x\in K}|f(x)-g(x)| < \epsilon$, where each $g_i(z)$ is either $\sin(z)$ or $\cos(z)$.

By Lemma 2, for each $g_i$ we may find
$$h_i(x) := \sum_{j=1}^{s} d_{ij}\,\sigma(\lambda_{ij}\, x + \theta_{ij})$$
such that for all $x\in w_i\cdot K$,
$$|g_i(x) - h_i(x)| < \frac{\epsilon}{|c_i|\cdot r}.$$
Define $h:\RR^n\to \RR$ by
$$h(x) = \sum_{i=1}^r c_i \, h_i(w_i \cdot x).$$
Observe that
$$h(x) = \sum_{i=1}^r c_i \sum_{j=1}^s d_{ij}\,\sigma\bigl(\lambda_{ij}\cdot(w_i\cdot x) + \theta_{ij}\bigr) \;\in\; \mathcal{M}(\sigma),$$
and
$$
\begin{aligned}
|f(x)-h(x)| &\leq |f(x) - g(x)| + |g(x) - h(x)| \\
&\leq \epsilon + \Bigl|\sum_{i=1}^r c_i \bigl(g_i(w_i\cdot x) - h_i(w_i\cdot x)\bigr)\Bigr| \\
&\leq \epsilon + \sum_{i=1}^r |c_i|\cdot \frac{\epsilon}{|c_i|\cdot r} = 2\epsilon.
\end{aligned}
$$
Taking $\epsilon \to 0$ gives the desired result.
</div>

Now we proceed to the proofs of the lemmas.

### Proof of Lemma 1

We shall exhibit the proof for $n=1$. The general case is extended naturally via the multidimensional Fejér theorem, as we'll see. But first we define key concepts from harmonic analysis. Let $f:[-L,L]\to \RR$ be a periodic function.

<div class="definition">
<span class="env-label">Definition</span>
The <strong>Fourier coefficients</strong> of $f$ are
$$\hat{f}(k) = \frac{1}{2L}\int_{-L}^{L} f(x)\, e^{-i\pi k x/L}\, dx, \qquad k\in\ZZ.$$
</div>

<div class="definition">
<span class="env-label">Definition</span>
The <strong>partial Fourier sum</strong> of order $N$ is
$$S_N(f)(x) = \sum_{k=-N}^{N} \hat{f}(k)\, e^{i\pi k x/L}.$$
</div>

<div class="definition">
<span class="env-label">Definition</span>
The <strong>Fejér mean</strong> of order $N$ is the average of the first $N+1$ partial Fourier sums:
$$\sigma_N(f)(x) = \frac{1}{N+1}\sum_{j=0}^{N} S_j(f)(x).$$
</div>

<div class="theorem">
<span class="env-label">Theorem (Fejér Uniform Convergence)</span>
Let $f$ be a continuous function periodic on $[-L,L]$. Then $\sigma_N(f)(x) \to f(x)$ uniformly on $[-L,L]$.
</div>

The Fejér theorem is a classical result in harmonic analysis. Although the partial Fourier sums $S_N(f)$ may not converge uniformly for an arbitrary continuous function, their Cesàro averages (the Fejér means $\sigma_N(f)$) always do. Intuitively, averaging smooths out the oscillatory overshoots (Gibbs-like phenomena) present in the partial sums, yielding a well-behaved approximation that converges uniformly to $f$. This makes the Fejér means a more robust tool for approximation than the raw Fourier partial sums. A complete proof can be found in Stein and Shakarchi [\[4\]](#stein2003).

Using this theorem, we are now able to prove Lemma 1. First, observe that convolving with the kernel
$$F_n(x) = \sum_{k=-n}^{n}\Bigl(1-\frac{|k|}{n+1}\Bigr) e^{i\pi k x/L}$$
gives:

$$
\begin{aligned}
(f * F_n)(x) &= \frac{1}{2L}\int_{-L}^L f(y)\sum_{k=-n}^{n}\Bigl(1-\frac{|k|}{n+1}\Bigr) e^{i\pi k(x-y)/L}\, dy \\
&= \sum_{k=-n}^{n} e^{i\pi k x/L} \cdot \frac{1}{2L}\int_{-L}^L f(y)\Bigl(1-\frac{|k|}{n+1}\Bigr) e^{-i\pi k y/L}\, dy \\
&= \sum_{k=-n}^{n} e^{i\pi k x/L} \Bigl(1-\frac{|k|}{n+1}\Bigr) \hat{f}(k) \\
&= \frac{1}{n+1}\sum_{k=0}^n S_k(f)(x) \\
&= \sigma_n(f)(x).
\end{aligned}
$$

Crucially:
$$\sigma_n(f)(x) = \sum_{k=-n}^{n} e^{i\pi k x/L}\Bigl(1-\frac{|k|}{n+1}\Bigr) \hat{f}(k).$$
Since $f$ is real-valued, $\overline{\hat{f}(k)} = \hat{f}(-k)$, and so
$$\sigma_n(f)(x) = \hat{f}(0) + \sum_{k=1}^{n}\Bigl(1-\frac{k}{n+1}\Bigr) \bigl[\hat{f}(k) e^{i\pi k x/L} + \hat{f}(-k) e^{-i\pi k x/L}\bigr],$$
which rearranges to
$$\sigma_n(f)(x) = \frac{a_0}{2} + \sum_{k=1}^{n}\Bigl(1-\frac{k}{n+1}\Bigr) \Bigl[a_k \cos\bigl(\tfrac{\pi k x}{L}\bigr) + b_k \sin\bigl(\tfrac{\pi k x}{L}\bigr)\Bigr],$$
where $a_k = \hat{f}(k) + \hat{f}(-k) \in \RR$ and $b_k = i\bigl(\hat{f}(k) - \hat{f}(-k)\bigr)\in \RR$.

Therefore $\sigma_n(f)(x) \in T_1$ for every $n$. We can now conclude Lemma 1 for $n=1$: given any $f\in C(\RR)$ and compact $K\subset \RR$, choose $[-L,L]$ strictly containing $K$. Continuously extend $f$ over $[-L,L]$ so that $f|_K$ is unchanged but $f(-L)=f(L)$. Then $f$ is periodic on $[-L,L]$, the sequence $\sigma_n(f)(x) \in T_1$ converges to $f$ uniformly on $[-L,L]$, and hence on $K$.

#### Generalization to $n>1$

Let $f\in C(\RR^n)$ and $K\subset \RR^n$ compact. Choose $L>0$ large enough that $K\subset (-L,L)^n$. As before, extend $f$ continuously to a function periodic on $[-L,L]^n$ (periodic in each coordinate with period $2L$) agreeing with $f$ on $K$.

The multivariate Fourier coefficients are
$$\hat{f}(k) = \frac{1}{(2L)^n}\int_{[-L,L]^n} f(x)\, e^{-i\pi (k\cdot x)/L}\, dx, \qquad k\in\ZZ^n,$$
and the multivariate Fejér mean of order $N$ is
$$\sigma_N(f)(x) = \sum_{\substack{k\in\ZZ^n \\ \|k\|_\infty \leq N}} \prod_{j=1}^{n}\Bigl(1-\frac{|k_j|}{N+1}\Bigr) \hat{f}(k)\, e^{i\pi (k\cdot x)/L}.$$
By the multidimensional Fejér theorem, $\sigma_N(f)\to f$ uniformly on $[-L,L]^n$.

By the same real-valuedness argument as the $n=1$ case (pairing each $k$ with $-k$ and using $\overline{\hat{f}(k)}=\hat{f}(-k)$), each $\sigma_N(f)(x)$ can be rewritten as a finite linear combination of terms $\cos\bigl(\tfrac{\pi k\cdot x}{L}\bigr)$ and $\sin\bigl(\tfrac{\pi k\cdot x}{L}\bigr)$ for $k\in\ZZ^n$. Since $\tfrac{\pi k}{L}\in\RR^n$, each such term belongs to $T_n$, and hence $\sigma_N(f)\in T_n$. The uniform convergence $\sigma_N(f)\to f$ on $K$ then gives density of $T_n$ in $C(\RR^n)$.

Having established that trigonometric polynomials are dense in $C(\RR^n)$, it remains to show that the network space $\mathcal{M}(\sigma)$ can approximate each trigonometric component. This is the content of Lemma 2, to which we now turn.

### Proof of Lemma 2

We rely on the following auxiliary result.

<div class="lemma">
<span class="env-label">Lemma 3</span>
Let $\sigma \in C(\RR)$ be a nonpolynomial function. Then there exists some $t_0\in \RR$ for which $\sigma^{(k)}(t_0) \neq 0$ for all $k\geq 0$.
</div>

A proof of Lemma 3 appears in Donoghue [\[5\]](#donoghue1969) (p. 53), though simpler proofs exist in the literature.

Using Lemma 3, observe first that for any $\lambda$, $f_{\lambda}(t) := \sigma(\lambda t + t_0) \in N_{\sigma}$. Hence we can form the sequence of functions
$$\left\{\frac{f_\lambda - f_0}{h}\right\}_{h\to 0} \subset N_{\sigma}.$$
We then have $\frac{d f_{\lambda}}{d\lambda}(0)\in \overline{N_{\sigma}}$. By the same argument, $\frac{d^k}{d\lambda^k}f_{\lambda}(0) \in \overline{N_{\sigma}}$ for all $k\geq 1$.

Now observe that
$$\frac{d^k}{d\lambda^k}f_{\lambda}(0) = t^k \, \sigma^{(k)}(t_0) = c_k\, t^k$$
for nonzero $c_k$ (by Lemma 3). Hence all monomials $t^k$ — and therefore all polynomials — lie in $\overline{N_{\sigma}}$.

To conclude, let $f\in C(\RR)$, $K\subset \RR$ compact, and $\epsilon > 0$. By the Stone–Weierstrass theorem, there exists a polynomial $p(t) = \sum_{j=0}^{r} a_j t^j$ such that $\sup_{t\in K} |f(t) - p(t)| < \epsilon$. Since each monomial $c_j t^j \in \overline{N_{\sigma}}$, for each term $a_j t^j$ we can find $q_j \in N_{\sigma}$ satisfying
$$\sup_{t\in K} |a_j t^j - q_j(t)| < \frac{\epsilon}{r+1}.$$
Setting $q = \sum_{j=0}^{r} q_j \in N_{\sigma}$, the triangle inequality gives
$$\sup_{t\in K} |f(t) - q(t)| \leq \sup_{t\in K} |f(t) - p(t)| + \sup_{t\in K} |p(t) - q(t)| < \epsilon + \sum_{j=0}^{r} \frac{\epsilon}{r+1} = 2\epsilon.$$
Since $\epsilon$ was arbitrary, $N_{\sigma}$ is dense in $C(\RR)$ as desired.

## Conclusion

We have presented a self-contained proof of the universal approximation theorem for single-hidden-layer neural networks with continuous, nonpolynomial activation functions. The argument was decomposed into two density results: Lemma 1, which establishes that trigonometric polynomials are dense in $C(\RR^n)$ via the multidimensional Fejér theorem, and Lemma 2, which shows that univariate translates of any continuous nonpolynomial activation are dense in $C(\RR)$ by recovering all monomials in the closure and invoking Stone–Weierstrass. Composing these two steps yields the classical result that $\mathcal{M}(\sigma)$ is dense in $C(\RR^n)$, providing a transparent and elementary alternative to the original functional-analytic proofs of Cybenko and Hornik.

---

<div class="bibliography">

**References**

<ol>
<li id="cybenko1989">G. Cybenko, <em>Approximation by superpositions of a sigmoidal function</em>, Mathematics of Control, Signals and Systems, 2(4):303–314, 1989. <a href="https://web.njit.edu/~usman/courses/cs675_fall18/10.1.1.441.7873.pdf" target="_blank" rel="noopener">PDF</a></li>
<li id="hornik1991">K. Hornik, <em>Approximation capabilities of multilayer feedforward networks</em>, Neural Networks, 4(2):251–257, 1991.</li>
<li id="survey2024">S. Kratsios et al., <em>Universal approximation theory: A survey</em>, arXiv:2407.12895, 2024. <a href="https://arxiv.org/abs/2407.12895" target="_blank" rel="noopener">arXiv</a></li>
<li id="stein2003">E. M. Stein and R. Shakarchi, <em>Fourier Analysis: An Introduction</em>, Princeton Lectures in Analysis, vol. 1, Princeton University Press, 2003.</li>
<li id="donoghue1969">W. F. Donoghue, <em>Distributions and Fourier Transforms</em>, Academic Press, New York, 1969.</li>
</ol>

</div>
