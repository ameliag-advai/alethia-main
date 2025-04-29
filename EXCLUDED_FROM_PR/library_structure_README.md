# Python Library Structure for Research Projects

This document explains the purpose of each subdirectory in the `advai` package of the project.

- **model**: Contains model objects used in testing. This directory interfaces with models stored in files or accessed through APIs.
- **data**: Stores PyTorch dataset objects that interface with the data in files.
- **preprocessing**: Contains code for transforming raw data from the dataset objects into a format that the models can consume (e.g., normalization).
- **artifact**: A collection of data, preprocessing, and model objects that can be tested or used as part of another test for another artifact.
- **procedure1, procedure2, procedure3**: Directories for procedures created to solve specific problems (e.g., Mahalanobis distance for OOD detection).
- **utils**: Stores utility functions that are used repeatedly throughout the project (e.g., a function for converting bounding box coordinates to the correct format).
- **research_evaluation**: Contains objects and functions that researchers develop to test and evaluate the procedures, showing that the procedures work as intended (e.g., demonstrate that Mahalanobis distance can recognize that cars are out-of-distribution for a model that classifies cats and dogs).

Feel free to modify this structure according to your project's specific needs.

