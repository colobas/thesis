---
title: ""
author: [Guilherme Pires]
date: 2018-06-28
subject: "Tese"
tags: [tese]
header-includes: |
  \usepackage{bm}
  \usepackage{amsfonts}
  \usepackage{amssymb}
---

\begin{align}
    \log\ p_\theta (\bm x) &\geq \int q_\phi (\bm{z|x})\ \log(\frac{p_\theta (\bm{x,z})}{q_\phi (\bm{z|x})}) \\
    &= \int q_\phi (\bm{z|x})\ \log(\frac{p_\theta (\bm{z|x}) p_\theta (\bm x)}{q_\phi (\bm{z|x})}) \\
    &= \int q_\phi (\bm{z|x})[\log(\frac{p_\theta (\bm{z|x})}{q_\phi (\bm{z|x})}) + \log\ p_\theta (\bm x)] \\
    &= \mathbb{E}_{q_\phi (\bm{z|x})}[\log\ p_\theta (\bm x)] - D_{KL}(q_\phi (\bm{z|x})\ \|\ p_\theta (\bm{z|x})) \\
    &= \log\ p_\theta (\bm x) - D_{KL}(q_\phi (\bm{z|x})\ \|\ p_\theta (\bm{z|x}))
\end{align}

Where the last step arises from the fact that the function inside the expectation doesn't depend on z.

---

\begin{align}
    p_\theta (\bm{x,z}) &= \prod_{t=1}^T p_\theta (x_t | \bm{z_{\leq t}, x_{<t}}) p_\theta(z_t | \bm{z_{<t}, x_{<t}})\\
\end{align}

---

\begin{align}
    p_\theta (\bm x) &= \int p_\theta (\bm{x,z})\ d\bm z \\
    &= \int \int ... \int p_\theta (\bm{x,z})\ dz_1 dz_2\ ...\ dz_T \\
    &= \int \int ... \int [\prod_{t=1}^T p_\theta (x_t | \bm{z_{\leq t}, x_{<t}}) p_\theta(z_t | \bm{z_{<t}, x_{<t}})]\ dz_1 dz_2\ ...\ dz_T \\
    &= \int \int ... \int ... \int p_\theta (x_2 | z_{1:2}, x_1) p_\theta (z_2 | z_1, x_1) \int p_\theta (x_1 | z_1) p_\theta (z_1) dz_1 dz_2\ ...\ dz_T \\
    &= \int \int ... \int ... \int p_\theta (x_2 | z_{1:2}, x_1) p_\theta (z_2 | z_1, x_1) \ \mathbb{E}_{p_\theta(z_1)}[p_\theta (x_1 | z_1)] dz_2\ ...\ dz_T \\
    &= \int \int ... \int p_\theta (x_3 | z_{1:3}, x_{1:2}) p_\theta (z_3 | z_{1:2}, x_{1:3})\ \mathbb{E}_{p_\theta(z_2 | x_1, z_1)}[p_\theta (x_2 | z_{1:2}, x_1)]\ \mathbb{E}_{p_\theta(z_1)}[p_\theta (x_1 | z_1)] dz_3\ ...\ dz_T \\
    &= \prod_{t=1}^T \mathbb{E}_{p_\theta(z_t|\bm{z_{<t}, x_{<t}})}[p_\theta (x_t |\bm{ z_{\leq t}, x_{<t} })]
\end{align}

---

\begin{align}
    \log\ p_\theta (\bm x) &= \log \prod_{t=1}^T \mathbb{E}_{p_\theta(z_t|\bm{z_{<t}, x_{<t}})}[p_\theta (x_t |\bm{ z_{\leq t}, x_{<t} })] \\
    &= \sum_{t=1}^T \log\ \mathbb{E}_{p_\theta(z_t|\bm{z_{<t}, x_{<t}})}[p_\theta (x_t |\bm{ z_{\leq t}, x_{<t} })] \\
    &\geq \sum_{t=1}^T \mathbb{E}_{p_\theta(z_t|\bm{z_{<t}, x_{<t}})}[\log\ p_\theta (x_t |\bm{ z_{\leq t}, x_{<t} })] \\
    &= \mathbb{E}_{p_\theta(z_t|\bm{z_{<t}, x_{<t}})}[\sum_{t=1}^T \log\ p_\theta (x_t |\bm{ z_{\leq t}, x_{<t} })]
\end{align}

---

So finally we get:

\begin{align}
    \log\ p_\theta (\bm x) &\geq \log\ p_\theta (\bm x) - D_{KL}(q_\phi (\bm{z|x})\ \|\ p_\theta (\bm{z|x})) \\
    &\geq \mathbb{E}_{p_\theta(z_t|\bm{z_{<t}, x_{<t}})}[\sum_{t=1}^T \log\ p_\theta (x_t |\bm{ z_{\leq t}, x_{<t} })] - D_{KL}(q_\phi (\bm{z|x})\ \|\ p_\theta (\bm{z|x})) \\
    &= \mathcal{L} (\theta, \phi)
\end{align}


