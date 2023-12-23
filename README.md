# Probabilistic-Graphical-Models-Project

This repository contains all the codes and scripts necessary to replicate the experiments and concepts discussed in our project report for the 2023/2024 M2 MVA course "Probabilistic Graphical Models and Deep Generative Models".

## Overview
The project explores score-based generative models, addressing key aspects such as:
- Score-matching techniques and their origins.
- Langevin dynamics for sampling and associated challenges.
- Addressing issues like the manifold hypothesis and low data density regions.
- Implementation and analysis of Denoising Score Matching (DSM).
- Experimental evaluations using noise schedules on datasets like MNIST, CIFAR-10, and the Stanford Car dataset, achieving good results.
- Comparative analysis using Fréchet Inception Distance (FID) scores.
- Theoretical connections between score-matching and diffusion models.

## Files in the Repository
- `main.py`: Includes usage examples for training and sampling.
- `compute_FID.py`: Script to compute FID scores for model evaluation.
- `Report.pdf`: Detailed project report explaining the theoretical background, methodologies, and experimental results.

## Getting Started
To get started with the project:
1. Clone the repository.
2. Install required dependencies.
3. Run `main.py` for training and sampling demonstrations.
4. Use `compute_FID.py` to evaluate your models.

## Authors
This project was developed by:
- [Alessio Spagnoletti]
- [Antoine Ratouchniak]
- [Dana Aubakirova]

## Contributions
This project is open for contributions. If you have suggestions or improvements, feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


