# Cyton2 modelling of dysregulated T cells from coeliac disease

This repository contains the datasets from human T-cell momentum assay and analysis codes to explore and estimate Cyton2 [[1]](#1) parameters. Due to large number of sample sizes (170 for CD4+ and 269 for CD8+), two identical scripts, "cyton2_CD4.py" and "cyton2_CD8.py", are provided to efficiently and simultaneously fit all of the donors in three different conditions: T cell medium (TCM) only; IL-2 blocking (Block, anti-IL-2/IL-2Ra); and, IL-2 blocking plus recombinant human IL-2 (Block+IL-2, anti-IL-2/IL-2Ra + rhuIL-2). Moreover, we offer a Jupyter notebook to visualise raw cell population dynamics as well as summary statistics based on cohort method [[2]](#2), which is available for open source software written in Java ([cohort-explorer](https://github.com/hodgkinlab/cohort-explorer)). The results of permutation tests, shown in the paper, are also located in the notebook.

All figures are located in $\verb_out_$ folder and organised by the cell type.

- $\verb_analysis_$ folder contains figures generated from the Jupyter notebook.
- $\verb_TCM_$, $\verb_Block_$ and $\verb_Block+IL-2_$ folders contain Cyton2 model fit results for each CD4+ and CD8+ samples.

# References
<a id="1">[1]</a>
H. Cheon, A. Kan, G. Prevedello, S. C. Oostindie, S. J. Dovedi, E. D. Hawkins, J. M. Marchingo, S. Heinzel, K. R. Duffy, P. D. Hodgkin, Cyton2: A Model of Immune Cell Population Dynamics That Includes Familial Instructional Inheritance. Frontiers Bioinform 1, 723337 (2021).

<a id="2">[2]</a>
E. D. Hawkins, M. Hommel, M. L. Turner, F. L. Battye, J. F. Markham, P. D. Hodgkin, Measuring lymphocyte proliferation, survival and differentiation using CFSE time-series data. Nat Protoc 2, 2057–2067 (2007).