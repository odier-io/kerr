########################################################################################################################

all: manual
	echo 'Ok.'

########################################################################################################################

.PHONY: manual
manual:
	cd manual && make html

########################################################################################################################

.PHONY: test
test:
	python3 -m pytest ./test/

########################################################################################################################

.PHONY: cov
cov:
	USE_NUMBA_CPU=0 python3 -m pytest --cov=kerr --cov-report xml:coverage.xml ./test/

########################################################################################################################

.PHONY: htmlcov
htmlcov:
	USE_NUMBA_CPU=0 python3 -m pytest --cov=kerr --cov-report html ./test/

########################################################################################################################

.PHONY: deploy
deploy: doc test
	./setup.py sdist && twine check dist/* && twine upload dist/*

########################################################################################################################

clean:
	rm -fr ./build/ ./*.egg-info/ ./.pytest_cache/ ./doc/_build/ ./coverage.xml

########################################################################################################################
