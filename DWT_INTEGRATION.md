# DWT integration

The existing GLCM/U-Net inference path is preserved. The DWT branch uses the same
current-image tumor ROI, crops it with five pixels of padding, normalizes to 0–1,
and applies level-one Haar DWT using PyWavelets. LL, LH, HL, HH each contribute
mean, standard deviation, variance, energy, and entropy, in notebook order.

## Saved artifacts and data

- `MODELS/final_svm_fusion_pca.pkl`: DWT SVM, confirmed by
  `NOTEBOOKS/Training_SVM_DWT.ipynb` (the filename says fusion, but its input is DWT PCA).
- `MODELS/pca_knn_dwt_final.joblib`: DWT KNN with 7 neighbors and 15 PCA inputs.
- `DATASET/Features_DWT/dwt_scaler.pkl`: fitted 20-feature StandardScaler.
- `DATASET/Features_DWT/dwt_pca.pkl`: fitted 20-to-15-component PCA.
- `DATASET/Features_DWT/dwt_{train,test}.csv`: raw DWT features and image identifiers.
- `DATASET/Features_DWT/dwt_pca_{train,test}.csv`: PC1–PC15 and image identifiers.
- `DATASET/Features(1)/glcm_{train,test}.csv` and `pca_{train,test}.csv`:
  existing GLCM and GLCM PCA reference tables.

The DWT manager loads artifacts on first use and validates both train/test CSV
column orders against the extractor and classifier. No fitting or retraining is
performed. Both DWT classifiers use the same saved scaler and PCA transformation.
Install `PyWavelets` from the updated requirements in the backend environment.

## Additive API and UI

`POST /api/predict` adds nullable `dwt_svm` and `dwt_knn`, using the existing classifier result
schema (`prediction`, `score`, `score_type`, `score_unit`, `probabilities`). Score
is the saved classifier's probability for its predicted class, expressed as a percentage.

Metadata adds `dwt_status`, `dwt_error`, `dwt_feature_names`,
`dwt_feature_values`, `dwt_pca_feature_names`, and `dwt_pca_components`.
DWT failures return null for both DWT classifier fields and an error message while retaining the
successful original results. The original empty-ROI 422 behavior is unchanged.

`GET /api/feature-tables/{pipeline}` supports `glcm`, `glcm_pca`, `dwt`, and
`dwt_pca`; parameters are `split=train|test`, `offset`, and `limit` (maximum 100).
It returns `source`, `split`, `total`, `columns`, and `rows`.

The frontend adds side-by-side DWT SVM and KNN classification cards and probability charts and current-scan DWT/DWT PCA tables.
The saved-dataset expander loads all four reference tables on request, with train/test
selection, paging, source filenames, and loading/error states. Reference rows are
explicitly distinguished from current-scan inference values.

## Verification

Run `python -m unittest discover -s tests -v` with the existing backend/frontend
dependencies and local models/datasets installed. All 10 tests passed locally:
real U-Net/API inference, GLCM prediction parity, DWT feature/PCA parity against
12 saved ROIs across both splits and all three classes, artifact/schema failure
handling, empty inputs, table pagination/content, and Streamlit AppTest rendering.
`python -m compileall -q BACKEND FRONTEND tests` also passed. No separate frontend
build, lint, or type-check configuration exists.

The GLCM diagnostic panel is expanded by default and displays all 96 raw features
and all 50 PCA coordinates. Its labels and API component count now reflect the
actual saved PCA output; its transformation and predictions are unchanged.
DWT uses its actual 15 components. The test run emitted existing
TensorFlow/Altair deprecation and Streamlit test-context warnings, without failures.

## Final ensemble

The additive `ensemble` response field combines the four existing percentage
probability dictionaries using Weighted Soft Voting. Class names are aligned
explicitly and rounding drift from the individual two-decimal API probabilities
is normalized. The largest combined probability determines prediction/confidence;
exact ties follow the configured class order. Existing classifier outputs stay unchanged.

Each classifier receives 25% weight. Notebook CV scores are not comparable:
GLCM/DWT SVM report macro-F1 (81.05%/74.79%), while GLCM/DWT KNN report accuracy
(75.02%/71.10%). In addition, the GLCM KNN notebook saves
`pca_knn_glcm_final.joblib`, whereas the app loads `pca_knn_glcm.joblib`.
There is no comparable validation score set tied to all four deployed artifacts;
held-out test scores were not used to select weights.

The final section is appended after all existing results and shows prediction,
confidence, three class probabilities, voting method, and individual predictions
and weights. Missing or invalid probabilities make only the ensemble unavailable;
no partial vote or substitute prediction is generated. Ensemble tests cover class
alignment, probability averaging, rounding, ties, invalid/missing inputs, rendering,
and comparison against the four outputs from a real MRI API request.
