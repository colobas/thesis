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
    p_\theta (\bm{x,z}) &= \prod_{t=1}^T p_\theta (x_t | \bm{z_{\leq t}, x_{<t}}) p_\theta(z_t | \bm{z_{<t}, x_{<t}}) \\
    &= \left(\prod_{t=1}^T p_\theta (x_t | \bm{z_{\leq t}, x_{<t}}) \right) \left(\prod_{t=1}^T p_\theta(z_t | \bm{z_{<t}, x_{<t}}) \right)
\end{align}

---

\begin{align}
    \log\ p_\theta (\bm x) &\geq \int q_\phi (\bm{z|x})\ \log(\frac{p_\theta (\bm{x,z})}{q_\phi (\bm{z|x})}) \\
    &= \int q_\phi (\bm{z|x})\ \log \left( 
        \frac{\left(\prod_{t=1}^T p_\theta (x_t | \bm{z_{\leq t}, x_{<t}} \right) \left(\prod_{t=1}^T p_\theta(z_t | \bm{z_{<t}, x_{<t}}) \right)}
             {q_\phi (\bm{z|x})}\right) \\
    &= \int q_\phi (\bm{z|x})\left[
        \log\left( \prod_{t=1}^T p_\theta (x_t | \bm{z_{\leq t}, x_{<t}}) \right) +
        \log\left(\frac{ \prod_{t=1}^T p_\theta(z_t | \bm{z_{<t}, x_{<t}})}{q_\phi (\bm{z|x})}\right) \right] \\
    &= \int q_\phi (\bm{z|x})\left[
        \left( \sum_{t=1}^T \log p_\theta (x_t | \bm{z_{\leq t}, x_{<t}}) \right) +
        \log\left(\frac{ p_\theta(\bm{z|x})}{q_\phi (\bm{z|x})}\right) \right] \\
    &= \mathbb{E}_{q_\phi (\bm{z|x})}[\sum_{t=1}^T \log p_\theta (x_t | \bm{z_{\leq t}, x_{<t}})] - D_{KL}(q_\phi (\bm{z|x})\ \|\ p_\theta (\bm{z|x})) \\
\end{align}

