### Machine learning reveals hidden diagnosis among underserved patients

Thanks for your interest in the code accompanying our work. Due to patient privacy restrictions, the data underlying the analysis cannot be made publicly available. However, we do provide access to the code used to generate the risk score and quantify diagnosis disparities. We share these materials in the interest of reproducibility and in support of future efforts to analyze clinical decision-making using machine learning.

The code is organized as follows: 
- `notebooks/` contains a sequence of notebooks used to produce all figures and tables provided in the paper. 
    - `step01_create_muse_edw_map.ipynb` collates information necessary to link patients from the EHR database to the ECG database.
    - `step02_process_muse_cache.ipynb` processes all ECGs in the ECG database to speed up later stages of the data generation process. 
    - `step10_generate_AF_data.ipynb` creates the dataset used to train the risk score and generate results. To quote the paper directly: "For patients with no observed AF, their first sinus rhythm ECG serves as a negative example. For patients with observed AF, we first identify the earliest ECG in which AF occurs, and retain patients who have a recorded sinus rhythm ECG in the 90 days prior to their AF ECG;  these patients serve as positive examples." We also pre-process the ECGs, as is standard, by removing baseline wander, truncating outliers, and normalizing the observed values.
    - `step20_summary_stats_AF.ipynb` generates Table S1 (summary statistics of the three samples) and Table 1 (diagnosis rates among patients with confirmed AF, across demographic groups)
    - `step30_train_AF_models.ipynb` conducts the main analysis. It loads the ECG dataset, merges in variables drawn from the EDW, trains a deep learning model to estimate a patient's AF risk, evaluates the risk score's performance, and estimates disparities in diagnosis conditional on risk across demographic groups. The notebook contains the code to produce Figures 2, 3, 4, S1, S2, S3, S4, S5, and S6 and Table 2.
- `src/` contains functions used to conduct our analysis. The files most relevant to reproducing results are `models.py`, which contains the model architecture, and `training.py`, which contains the code used to learn the deep learning model's weights.
