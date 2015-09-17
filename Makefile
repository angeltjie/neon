# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Top-level control of the building/installation/cleaning of various targets
#
# set empty to prevent any implicit rules from firing.
.SUFFIXES:

# where our installed python packages will live
VIRTUALENV_DIR := .venv
VIRTUALENV_EXE := virtualenv  # use pyvenv for python3 install
ACTIVATE := $(VIRTUALENV_DIR)/bin/activate

# get release version info
RELEASE := $(strip $(shell grep '^VERSION *=' setup.py | cut -f 2 -d '=' \
	                         | tr -d "\'"))

# basic check to see if any CUDA compatible GPU is installed
# set this to false to turn off GPU related functionality
HAS_GPU := $(shell nvcc --version > /dev/null 2>&1 && echo true)

# set this to true to install visualization dependencies and functionality
# (off by default)
VIS :=

# style checking related
STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS := neon bin tests

# pytest options
TEST_OPTS :=

# arguments to running examples
EXAMPLE_ARGS := -e1

# this variable controls where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

# Maxwell assembler project related
MAXAS_SRC_URL := https://github.com/NervanaSystems/maxas.git
MAXAS_DL_DIR := $(VIRTUALENV_DIR)/maxas
MAXAS := $(VIRTUALENV_DIR)/bin/maxas.pl
MAXAS_PLIB := PERL5LIB=$(VIRTUALENV_DIR)/maxas/lib

# GPU Kernel compilation related
KERNEL_BUILDER := neon/backends/make_kernels.py
KERNEL_BUILDER_BUILD_OPTS := --kernels
KERNEL_BUILDER_CLEAN_OPTS := --clean

# neon compiled objects
IMAGESET_DECODER := neon/data/imageset_decoder.so

.PHONY: default env maxas kernels sysinstall sysuninstall clean_py clean_maxas \
	      clean_so clean_kernels clean test coverage style lint check \
	      doc html release examples serialize_check

default: env

env: $(ACTIVATE) kernels $(IMAGESET_DECODER)

$(ACTIVATE): requirements.txt gpu_requirements.txt vis_requirements.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install -U pip
	@# cython added separately due to h5py dependency ordering bug.  See:
	@# https://github.com/h5py/h5py/issues/535
	@. $(ACTIVATE); pip install cython==0.23.1
	@. $(ACTIVATE); pip install -r requirements.txt
ifeq ($(VIS), true)
	@echo "Updating visualization related dependecies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r vis_requirements.txt
endif
	@echo
ifeq ($(HAS_GPU), true)
	@echo "Updating GPU dependencies in $(VIRTUALENV_DIR)..."
	@. $(ACTIVATE); pip install -r gpu_requirements.txt
	@echo
endif
	@echo "Installing neon in development mode..."
	@. $(ACTIVATE); python setup.py develop
	@echo "######################"
	@echo "Setup complete.  Type:"
	@echo "    . '$(ACTIVATE)'"
	@echo "to work interactively"
	@echo "######################"
	@touch $(ACTIVATE)
	@echo

maxas: $(ACTIVATE) $(MAXAS_DL_DIR)
ifeq ($(HAS_GPU), true)
	@cd $(MAXAS_DL_DIR) && git pull >/dev/null 2>&1
	@test -f $(MAXAS) ||\
		{ echo "Installing maxas..." &&\
		  ln -s ../maxas/bin/maxas.pl $(MAXAS) ;\
		  echo "";\
		}
endif

$(MAXAS_DL_DIR):
ifeq ($(HAS_GPU), true)
	@test -d $(MAXAS_DL_DIR) ||\
		{ echo "Cloning maxas repo..." ;\
		  git clone $(MAXAS_SRC_URL) $(MAXAS_DL_DIR) ;\
		  echo "";\
		}
endif

kernels: $(ACTIVATE) maxas
ifeq ($(HAS_GPU), true)
	@. $(ACTIVATE); $(MAXAS_PLIB) $(KERNEL_BUILDER) $(KERNEL_BUILDER_BUILD_OPTS)
	@echo
endif

$(IMAGESET_DECODER): $(subst so,cpp,$(IMAGESET_DECODER))
ifeq ($(shell pkg-config --modversion opencv >/dev/null 2>&1; echo $$?), 0)
	@echo "Compiling $(IMAGESET_DECODER) ..."
  ifeq ($(shell uname -s), Darwin)
		-g++ -w -O3 -stdlib=libc++ -shared -o $(IMAGESET_DECODER) -std=c++11 -fPIC $< $$(pkg-config opencv --cflags --libs)
  else
		-g++ -w -O3 -shared -o $(IMAGESET_DECODER) -std=c++11 -fPIC $< $$(pkg-config opencv --cflags --libs)
  endif
else
	@echo "pkg-config or opencv not installed.  Unable to build imageset_decoder"
	@echo
endif

# TODO: remove env dep and handle kernel/.so compilation via setup.py directly
sysinstall: env
	@echo "Installing neon system wide..."
	@pip install -U pip
	@# cython added separately due to h5py dependency ordering bug.  See:
	@# https://github.com/h5py/h5py/issues/535
	@pip install cython==0.23.1
	@pip install -r requirements.txt
ifeq ($(VIS), true)
	@pip install -r vis_requirements.txt
endif
ifeq ($(HAS_GPU), true)
	@pip install -r gpu_requirements.txt
endif
	@pip install .
	@echo

sysuninstall:
	@echo "Uninstalling neon system wide..."
	@pip uninstall neon
	@echo

clean_py:
	@echo "Cleaning compiled python object files..."
	@find . -name "*.py[co]" -type f -delete
	@echo

clean_so:
	@echo "Cleaning compiled shared object files..."
	@rm -f $(IMAGESET_DECODER)
	@echo

clean_maxas:
ifeq ($(HAS_GPU), true)
	@echo "Cleaning maxas installation and repo files..."
	@rm -f $(MAXAS)
	@rm -rf $(MAXAS_DL_DIR)
	@echo
endif

clean_kernels:
ifeq ($(HAS_GPU), true)
	@echo "Cleaning compiled gpu kernel files..."
	@test -f $(ACTIVATE) && . $(ACTIVATE); $(KERNEL_BUILDER) $(KERNEL_BUILDER_CLEAN_OPTS)
	@echo
endif

clean: clean_py clean_so clean_maxas clean_kernels
	@echo "Removing virtual environment files..."
	@rm -rf $(VIRTUALENV_DIR)
	@echo

test: env
	@echo "Running unit tests..."
	@. $(ACTIVATE); py.test $(TEST_OPTS) tests/ neon/backends/tests/
	@echo

examples: env
	@echo "Running all examples..."
	@. $(ACTIVATE); \
		for fn in `ls -1 examples/*.py`; \
		do \
		    echo "Running $$fn $(EXAMPLE_ARGS)"; \
		    python $$fn $(EXAMPLE_ARGS); \
			if [ $$? -ne 0 ]; \
	        then \
	            exit 1; \
			fi; \
		done;
	@echo

serialize_check: env
	@echo "Running CPU backend test of model serialization"
	@. $(ACTIVATE); python tests/serialization_check.py -e 10 -b cpu
	@echo

coverage: env
	@. $(ACTIVATE); py.test --cov=neon tests/ neon/backends/tests/
	@echo

style: env
	@. $(ACTIVATE); flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	@echo

lint: env
	@. $(ACTIVATE); pylint --output-format=colorized neon
	@echo

check: env
	@echo "Running style checks.  Number of style errors is... "
	-@. $(ACTIVATE); flake8 --count $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS) \
	                 > /dev/null
	@echo
	@echo "Number of missing docstrings is..."
	-@. $(ACTIVATE); pylint --disable=all --enable=missing-docstring -r n \
	                 neon | grep "^C" | wc -l
	@echo
	@echo "Running unit tests..."
	-@. $(ACTIVATE); py.test tests/ | tail -1 | cut -f 2,3 -d ' '
	@echo

doc: env
	@. $(ACTIVATE); neon --help > doc/source/neon_help_output.txt
	$(MAKE) -C $(DOC_DIR) clean
	@. $(ACTIVATE); $(MAKE) -C $(DOC_DIR) html
	@echo "Documentation built in $(DOC_DIR)/build/html"
	@echo

html: doc
	@echo "To view documents open your browser to: http://localhost:8000"
	@cd $(DOC_DIR)/build/html; python -m SimpleHTTPServer
	@echo

publish_doc: doc
ifneq (, $(DOC_PUB_HOST))
	@echo "relpath: $(DOC_PUB_RELEASE_PATH)"
	@-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh --perms --chmod=ugo+rX . \
		$(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_RELEASE_PATH)
	@-ssh $(DOC_PUB_USER)@$(DOC_PUB_HOST) \
		'rm -f $(DOC_PUB_PATH)/latest && \
		 ln -sf $(DOC_PUB_RELEASE_PATH) $(DOC_PUB_PATH)/latest'
else
	@echo "Can't publish.  Ensure DOC_PUB_HOST, DOC_PUB_USER, DOC_PUB_PATH set"
endif

dist: env
	@echo "Prepping distribution..."
	@python setup.py sdist

release: check dist
	@echo "Bump version number in setup.py"
	@vi setup.py
	@echo "Update ChangeLog"
	@vi ChangeLog
	@echo "TODO: commit changes"
	@echo "TODO: publish release to PYPI"
	@echo "TODO (manual script): publish documentation"
	@echo
