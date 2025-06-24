                                                                                                  Rating_prediction_engine

Project Overview:
This repository contains the implementation of a prototype recommendation system developed during my research internship at NIT Warangal. The goal was to model user-item interactions and accurately predict user ratings—an essential component in building scalable and intelligent recommendation systems used in e-commerce and media platforms.

Key Contributions:
Implemented a custom DataSplitter class to divide data into training, validation, and test sets while maintaining user and item distribution integrity.

Engineered user interaction history matrices using NumPy and PyTorch to represent structured input.

Developed a simple feedforward neural network in PyTorch for predicting user ratings on unseen items.

Utilized custom DataLoaders, GPU acceleration, and modular training/evaluation loops with MSE as the loss metric.

Created a sparse interaction matrix to better manage memory and computation for large datasets.

Modularized the entire pipeline to support future research with advanced model architectures.

Outcome:
The model was able to learn user preferences with good accuracy and generalization. The clean structure of the codebase supports further experimentation in recommender system research and development.

