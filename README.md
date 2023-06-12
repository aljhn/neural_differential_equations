# Neural Differential Equations

[JAX](https://github.com/google/jax) implementations of various neural differential equation architectures from scratch.
[Diffrax](https://github.com/patrick-kidger/diffrax) is used for handling the numerical integration.

## Neural Ordinary Differential Equations

Original paper by [Chen et al. \[2018\]](https://arxiv.org/abs/1806.07366).

Diffrax has an existing implementation of the adjoint method, but this part of Diffrax is not used,
only the numerical integration as it supports PyTrees unlike SciPy which only supports numpy arrays.

## Augmented Neural ODEs

[Dupont et al. \[2019\]](https://arxiv.org/abs/1904.01681).

## On Second Order Behaviour in Augmented Neural ODEs

[Norcliffe et al. \[2020\]](https://arxiv.org/abs/2006.07220).

## Dissecting Neural ODEs

[Massaroli et al. \[2020\]](https://arxiv.org/abs/2002.08071).

## Stable Neural Flows

[Massaroli et al. \[2020\]](https://arxiv.org/abs/2003.08063).

## Hamiltonian Neural Networks

[Greydanus et al. \[2019\]](https://arxiv.org/abs/1906.01563).

## Lagrangian Neural Networks

[Cranmer et al. \[2020\]](https://arxiv.org/abs/2003.04630).
