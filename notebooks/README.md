# Notes

This folder contains the notebooks to plot results in paper
```bibtex
Toward Interpretable Speech Deepfake Detection using Artifact-Specific Experts and Calibrated Detection Scores
accepted by ACM MM DFF-Workshop, 2026
```


## Dependnecy 

Basic Python tools: pandas, numpy, sklearn, matplotlib

## Structure

* `plot_paper_results`: **main notebook to plot the results in paper**. It is self-contained. 
* `plot_ensemble_results`: original version of the notebook above
* `plot_feat_examples`: plotting example figures in Figure 3
* `results`: copied from ~/v-viola/moe_antideepfake/results (Ansible). Results files needed to plot the figures/tables
  * `results/Ensemble/asv19/experts_trained_on_mix/CMLG/6_experts` was based on Viola's experiment after internship
  * `results/Experts` were added after running `main/example_*.py` that saves the uncalibrated scores

## How to use
* Run jupyternotebook and open `plot_paper_results`
* Run from the first block to the end black. Results are displayed in the notebook