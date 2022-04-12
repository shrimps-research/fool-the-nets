VENV := torch_library
DEFAULT_SERVER := doris

.ONESHELL:
SHELL := /bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

conda-env:
	conda create --name $(VENV) python
	$(CONDA_ACTIVATE) $(VENV)
	conda install cudnn cudatoolkit

activate-conda-env:
	$(CONDA_ACTIVATE) $(VENV)
	echo $$(which python)

install requirements.txt:
	$(CONDA_ACTIVATE) $(VENV)
	python -m pip install -r requirements.txt
	@echo export PYTHONPATH=$(PWD)/src >> ~/.bashrc
	bash

machine=$(DEFAULT_SERVER)
deploy $(id) $(machine):
	rsync -azv -e 'ssh -A -J $(id)@ssh.liacs.nl' ../fool-the-nets shrimps@$(machine):~/ --exclude-from='deployment-excluded-paths.txt'

download $(id) $(machine):
	rsync -azv -e 'ssh -A -J $(id)@ssh.liacs.nl' shrimps@doris:~/fool-the-nets/* ./fool-the-nets-'$(machine)'/

jupyter-doris $(id):
	ssh -N -f -L 9001:localhost:9001 $(id)@ssh.liacs.nl ssh -4 -N -f -L 9001:doris:9000 shrimps@doris

jupyter-boris $(id):
	ssh -N -f -L 8999:localhost:8999 $(id)@ssh.liacs.nl ssh -4 -N -f -L 8999:boris:9000 shrimps@boris
