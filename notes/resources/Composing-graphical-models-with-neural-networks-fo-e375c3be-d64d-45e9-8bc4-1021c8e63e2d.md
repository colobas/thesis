# Composing graphical models with neural networks for structured representation and fast inference

- Starting point
    - SVAE (Johnson et al)
        - [https://arxiv.org/abs/1603.06277](https://arxiv.org/abs/1603.06277)
        - [http://www.cs.toronto.edu/~duvenaud/talks/svae-slides.pdf](http://www.cs.toronto.edu/~duvenaud/talks/svae-slides.pdf)
    - rSLDS (Linderman et al)
        - [https://arxiv.org/pdf/1610.08466v1.pdf](https://arxiv.org/pdf/1610.08466v1.pdf)

- The initial idea was to apply the SVAE framework, having rSLDS as the latent PGM
- However rSLDS doesn't have a conjugate structure
- Some extensions of SVAE to non conjungate PGMs have been proposed
- Some similar frameworks have been proposed where the inference work-horse is different from the one in SVAE