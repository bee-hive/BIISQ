# BIISQ
Bayesian nonparametric discovery of Isoforms and Individual Specific Quantification with BIISQ


# Installation 
Requires python version 2 (2.7+)

## installation of required packages

Required packages: joblib cython numpy scipy

### for an existing installation of python/anaconda

If you have a working anaconda or python installation, then:

```
pip install --user numpy
pip install --user scipy
pip install --user joblib
pip install --user cython
```

### for a new installation of Python 2 (2.7+)

If you would like to install via a new vitual environment, then:

```
VIRT_HOME=/n/fs/biisq/virtenvs
cd $VIRT_HOME
virtualenv biisq_env
source biisq_env/bin/activate

pip install numpy
pip install scipy
pip install cython
pip install joblib
```

## installation of BIISQ

```
git clone https://github.com/bee-hive/BIISQ.git
cd BIISQ/examples/ex_gene_1/
sh ex_gene_1_run.sh
cd viz
sh run_viz.sh
```
there is another example for paired end data in BIISQ/examples/paired_end.

Running this script will summarize transcript composition and quantification.
By default results will be stored in BIISQ/examples/ex_gene_1/output.
The summary.html file contains descriptions of the individuals and isoform compositions.


# Running recommendations

If your data is very small, you might consider running BIISQ several times without proposals and merging, e.g. --max-n-prop 0 --min-n-prop 0

