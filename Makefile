# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
.PHONY: clean clean-build clean-pyc clean-docs clean-test clean-notebooks docs lint test coverage release dist install install-dev install-dev-deps help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
PIP_INSTALL := pip install --extra-index-url https://pypi.nvidia.com

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs clean-notebooks ## remove all build, test, docs, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -not -path "./.cache/*" -exec rm -fr {} +
	find . -name '*.egg' -not -path "./.cache/*" -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -not -path "./.cache/*" -exec rm -f {} +
	find . -name '*.pyo' -not -path "./.cache/*" -exec rm -f {} +
	find . -name '*~' -not -path "./.cache/*" -exec rm -f {} +
	find . -name '__pycache__' -not -path "./.cache/*" -exec rm -fr {} +

clean-docs: ## remove test and coverage artifacts
	rm -rf site

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .pytype/
	rm -fr .coverage*
	rm -f *-report.xml

clean-notebooks: ## remove Jupyter notebook output cells
	@echo "Cleaning notebook outputs..."
	find . -name '*.ipynb' -not -path "./.cache/*" -exec jupyter nbconvert --clear-output --inplace {} \;

docs: # TBD: uncomment after vdr clean-docs ## generate site
	cp CHANGELOG.md docs
	cp CONTRIBUTING.md docs
	cp LICENSE docs/LICENSE.md
	@mkdir -p docs/examples
	cp examples/README.md docs/examples/examples.md
	# Copy example subdirectory READMEs with directory structure
	@find examples -maxdepth 2 -mindepth 2 -name 'README.md' -type f | while read -r file; do \
		subdir=$$(dirname "$$file" | sed 's|^examples/||'); \
		mkdir -p "docs/examples/$$subdir"; \
		cp "$$file" "docs/examples/$$subdir/README.md"; \
	done

	mkdocs build --clean

docs-serve: docs
	mkdocs serve


lint: ## check style with pre-commit and pytype
	pre-commit run --all-files
	pytype aitune tests -j auto


test: ## run tests on
	pytest


all-tests: ## run all tests for all python versions
	tox --develop --skip-missing-interpreters


coverage: ## check code coverage quickly with the default Python
	coverage run --source aitune -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

dist: clean ## builds source and wheel package
	python3 -m build .
	ls -lh dist

install: clean ## install the package to the active Python's site-packages
	$(PIP_INSTALL) --upgrade pip
	$(PIP_INSTALL) .

install-dev: clean-build clean-pyc clean-test
	$(PIP_INSTALL) --upgrade pip
	$(PIP_INSTALL) -e ".[dev]"

-include *.mk
