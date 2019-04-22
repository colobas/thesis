# outline

# Representation learning of multivariate time-series

## (with applications in predictive maintenance, counterfactual inference, and global-individual learning)

- Main goal: learn interpretable latent representations of multivariate time-series
    - Ideally tackling problems such as missing data, unevenly sampled time-series and misaligned time-series
        - Can this be done using Neural ODEs?
    - Why/why not SSM?
    - Why/why not (V)AE?
- Sub goals:
    - Use latent representations to foresee hazards
        - Can hazards be circumscribed to certain regions of the latent space?
    - Use latent representations to ask "what if" questions (counterfactual inference)
        - This can be done by forward sampling whilst manipulating input features
    - Use latent representations to go from global models to asset specific models (maybe)
        - For example: asset specific GPs on top of globally learned representations

---

![](Untitled-5a460c52-c040-4cd3-a9be-41491779ccc6.png)

This but, make the last distribution autoregressive, as in the rSLDS paper