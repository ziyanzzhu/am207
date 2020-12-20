# A Systematic Comparison of Bayesian Deep Learning Robustness in Diabetic Retinopathy Tasks

## File Descriptions
- `writeup_final.ipynb`: summary and evaluation of the paper, numerical experiments and analysis
- `baselines.py`: implementation of four baselines / models: Deterministic, Deep Ensemble, MC Dropout, and Mean-field variational inference
- `uncertainty.py`: implementation of predictive entropy and the proposed uncertainty metric
- `utils.py`: plotting and test data generation scripts

Note: because some training takes a long time (especially for Deep Ensemble(, we saved some of the NN weights in the folder `/data` and load them directly if the data file exists. 

## Group Members
Ziyan Zhu, Nayantara Mudur, Felipe Gomez, Blake Duschatko
