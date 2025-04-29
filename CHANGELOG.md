# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2023-05-12

### Added:
- Damian's changes:
    - library_structure_README.md
    - research_evaluation subpackage
    - utils
    - template_notebook.ipynb

### Changed:
- Readme
- Makefile

## [0.3.0] - 2023-04-28

### Added:
- noteboks/templates/template_notebook.ipynb

### Changed:
- default-env to res-env

## [0.2.0] - 2023-04-18

### Added:
- Dockerfile
- docker-compose.yml
- default-env.yaml
- Makefile
- TODO
- notebooks directory (for templates, instructional, educational, demo, research, etc.):
    - templates/agnostic_template.ipynb
- advai namespace package directories:
    - artifact_catalogue
    - base
    - common
    - data (& core, tf, torch subpackages)
    - models (& core, tf, torch subpackages)
    - optimizers (& core, tf, torch subpackages)
    - preprocessing (& core, tf, torch subpackages)
    - procedures:
        - bias
        - drift
        - outlier_detection
        - outofdistribution
        - outofsample
        - poisoning
        - smallperturbation
        - training
    - task_catalouge

### Changed:
- Changelog
- Readme

### Removed:
- src/advai/template/template_class.py
- test/advai/unit/test_template_class.py
## [0.1.0] - 2023-02-28

### Added:
- .gitignore:
    - new files to be ignored:
        - *.pyc
        - *.pkl
        - *.pickle
        - *.csv
- requirements.txt;
- python_cicd.yaml:
    - pull request 'opened' trigger

### Changed:
- READMEmd:
    - python3.8+ badge to python3.8, python3.9, python3.10

### Removed:
- python_cicd.yaml:
    - pull request 'synchronize' trigger

## [0.0.0] - 2022-12-22
### Added
- Standard Python Repo Layout:
    - workflows
    - dir structure:
        - src/...
        - test/...
    - template package
    - TemplateClass
    - TemplateClass unit tests
    - .gitignore
    - CHANGELOG.md
    - LICENSE.txt
    - pyproject.toml
    - README.md
    - requirements.txt
    - VERSION
