# Cyton2 modelling of dysregulated T cells from coeliac disease

This repository contains the datasets from human T-cell momentum assay and analysis codes to explore and estimate [Cyton2](https://www.frontiersin.org/articles/10.3389/fbinf.2021.723337/full) parameters. Due to large number of sample sizes (170 for CD4+ and 269 for CD8+), two identical scripts, "cyton2_CD4.py" and "cyton2_CD8.py", are provided to efficiently and simultaneously fit all of the donors in three different conditions: T cell medium (TCM) only; IL-2 blocking (Block, anti-IL-2/IL-2Ra); and, IL-2 blocking plus recombinant human IL-2 (Block+IL-2, anti-IL-2/IL-2Ra + rhuIL-2\). Moreover, we offer a Jupyter notebook to visualise raw cell population dynamics as well as summary statistics based on [cohort method](https://www.nature.com/articles/nprot.2007.297), which is implemented in [this](https://github.com/hodgkinlab/cohort-explorer) open source software written in Java. The results of permutation tests, shown in the paper, are also located in the notebook.

All figures are located in $\verb+out+$ folder and organised by the cell type.

- $\verb+analysis+$ folder contains figures generated from the Jupyter notebook.
- $\verb+*-TCM+$, $\verb+*-Block+$ and $\verb_*-Block + IL-2_$ folders contain Cyton2 model fit results for each CD4+ and CD8+ samples.
