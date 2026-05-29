Installation
============

`uv <https://docs.astral.sh/uv/getting-started/installation/>`_ is recommended for installation:

.. code-block:: bash

    uv pip install fiqci-ems
    # or
    uv add fiqci-ems

Requires Python 3.11 or 3.12.

Verifying the installation
--------------------------

To test your installation clone
the repository and run the tests against the installed package.

.. code-block:: bash

    git clone https://github.com/fiqci/fiqci-ems.git
    cd fiqci-ems
    uv run --group dev pytest


HPC systems (Apptainer / Singularity)
-------------------------------------

On shared HPC systems it is recommended to install Python environments inside a
container using Apptainer/Singularity. The definition file below builds on the
official `uv <https://docs.astral.sh/uv/guides/integration/docker/>`_ image, so
``uv`` is already available in the container.

Save it as ``fiqci-ems.def``:

.. code-block:: text

    Bootstrap: docker
    From: ghcr.io/astral-sh/uv:python3.12-bookworm-slim

    %post
        uv pip install --system --no-cache fiqci-ems

And to build the image

.. code-block:: bash

    apptainer build --fakeroot fiqci-ems.sif fiqci-ems.def
    apptainer exec fiqci-ems.sif python3 my_script.py



LUMI
~~~~

On `LUMI <https://docs.lumi-supercomputer.eu/>`_ `fiqci-ems` comes pre-installed in
the `fiqci-vtt-qiskit` module which can be used via

.. code-block:: bash

    module load Local-quantum
    module load fiqci-vtt-qiskit

Otherwise you can use it in your own containers. It is highly recommended to
use one of the two container-based workflows below. See the LUMI documentation
on `installing Python <https://docs.lumi-supercomputer.eu/software/installing/python/#use-the-lumi-container-wrapper>`_,
the `container wrapper <https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/>`_
and `Singularity/Apptainer <https://docs.lumi-supercomputer.eu/software/containers/singularity/>`_
for background.

Define the environment once in an ``env.yml`` file.

.. code-block:: yaml

    channels:
      - conda-forge
    dependencies:
      - python=3.12
      - pip
      - pip:
          - fiqci-ems
          - pytest

Option 1: LUMI container wrapper (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``lumi-container-wrapper`` builds a container behind the scenes but exposes
the executables (``python``, ``pip``, ...) on your ``PATH`` as if the
environment were activated normally. This is the simplest option when you just
want to run Python scripts.

.. code-block:: bash

    module load LUMI
    module load lumi-container-wrapper

    mkdir -p $PWD/fiqci-ems
    conda-containerize new --prefix $PWD/fiqci-ems env.yml

    export PATH="$PWD/fiqci-ems/bin:$PATH"

To add or change packages later, use a post-install script:

.. code-block:: bash

    conda-containerize update $HOME/fiqci-ems --post-install post.sh

where ``post.sh`` contains commands run inside the environment, e.g.
``pip install <package>``.

Verify the installation on a compute node via ``srun``, from a clone of the
repository so the ``tests`` directory is available:

.. code-block:: bash

    git clone https://github.com/fiqci/fiqci-ems.git
    cd fiqci-ems
    srun --account project_xxx --partition interactive python3 -m pytest tests/


Option 2: Build a SIF image with cotainr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer an explicit, portable ``.sif`` image (for example to share it or
manage container execution yourself), build it with ``cotainr``:

.. code-block:: bash

    module load CrayEnv
    module load cotainr

    cotainr build fiqci-ems.sif --system=lumi-c --conda-env=env.yml

Verify the installation on a compute node via ``srun``.

.. code-block:: bash

    git clone https://github.com/fiqci/fiqci-ems.git
    srun --account=<project> --partition=interactive \
        singularity exec -B "$PWD:/mnt" fiqci-ems.sif \
        python3 -m pytest /mnt/fiqci-ems/tests -p no:cacheprovider
