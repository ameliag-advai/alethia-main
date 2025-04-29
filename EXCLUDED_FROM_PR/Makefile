.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

env:
	conda env create -f "./re.yaml" --name "re" --force --quiet
	$(CONDA_ACTIVATE) re
