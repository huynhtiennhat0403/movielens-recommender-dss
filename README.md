## MovieLens NCF recommender

The official experiment is an implicit-feedback pipeline based on Xiangnan He et al., “Neural Collaborative Filtering”. Every observed MovieLens rating is a positive interaction; negatives are sampled only from unseen movies.

The final comparison is **MF vs NCF**. GMF and MLP are independently pretrained branches. NCF copies their weights, combines the pretrained output layers with `alpha=0.5`, and is fine-tuned with SGD. Training uses dynamic 1:4 negative sampling, while evaluation uses the same deterministic 1-positive + 100-negative candidates for both models.

Run notebooks in this order:

1. `01_eda_preprocessing.ipynb`
2. `02_implicit_data_preparation.ipynb`
3. `03_matrix_factorization.ipynb`
4. `04_gmf_pretraining.ipynb`
5. `05_mlp_pretraining.ipynb`
6. `06_ncf_pretrained_training.ipynb`
7. `07_compare_mf_ncf.ipynb`

The backend loads `models/ncf_best.pth` and keeps raw ratings for user history and temporary user-vector adaptation.
