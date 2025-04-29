# Alethia Local Analysis

This repository contains the local analysis script and minimal required files for running patient diagnosis analysis with transformer-based models and sparse autoencoders.

## Contents
- `notebooks/local_analysis.py` — Main analysis script
- `release_conditions.json` — Minimal conditions metadata (required)
- `release_test_patients_mini_version.txt` — Small, shareable test dataset (required)
- `.env.example` — Template for required environment variables
- `.gitignore` — Ignores sensitive and large files

## Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd alethia-main
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ and install required packages (see below).
   ```sh
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install: `torch`, `transformer_lens`, `sae_lens`, `pandas`, `plotly`, `tqdm`, `python-dotenv`)*

3. **Set up your Hugging Face token:**
   - Copy `.env.example` to `.env` and add your token:
     ```sh
     cp .env.example .env
     # Edit .env and add your HF_TOKEN=...
     ```

4. **Prepare data files:**
   - Ensure `release_conditions.json` and `release_test_patients_mini_version.txt` are in the project root.
   - The script will use these files by default. Do NOT include large datasets in the repo.

## Usage

Run the analysis:
```sh
python notebooks/local_analysis.py
```

## Notes
- **Do NOT commit your `.env` file or any large/sensitive data!**
- If you need the full DDXPlus datasets, download them separately and follow the folder structure in the script, but do not add them to git.
- For questions or contributions, open an issue or pull request.

---

*This project is designed for secure, reproducible, and shareable local analysis. For more details, see the code comments in `local_analysis.py`.*
