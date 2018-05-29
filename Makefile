MODULE_PATHS=`python -c "import os, sys; print(' '.join('{}'.format(d) for d in sys.path if os.path.isdir(d)))"`
PIP=`which pip`

install:
	@echo "Installing via $(PIP)"
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install --upgrade --no-cache-dir \
	    scikit-learn pandas numpy scipy cython 	\
	    matplotlib bokeh jupyterlab 		\
	    pytest pylint sphinx

clean:
	find . -type f -name \*.pyc -delete
	find . -type f -name \*.pyo -delete
	find . -type d -name __pycache__ -delete
	rm -rf .ipynb_checkpoints/

tags:
	ctags -R --fields=+l --extra=+f --languages=python --python-kinds=-iv -f ./tags $(MODULE_PATHS)
