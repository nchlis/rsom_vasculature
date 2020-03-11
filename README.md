# rsom_vasculature
Calculating vasculature in RSOM images with machine learning

The following files are provided:

1. **step1_UNET_train.py**: trains a UNET (per patient/volunteer) on RSOM data.
2. **step2_UNET_predict.py**: predicts the epidermis and dermis regions using the trained UNETs.
3. **step3_UNET_postprocess.py**: Post-processes the UNET-predicted masks.
4. **step4_thresholding.py**: Predicts finer vasculature masks and features using traditional computer vision.
5. **step5_extract_features.py**: Extracts features and computes the vasculature score.

Available folders:
- **./trained_models**: will contain trained UNET models in the future.
- ** ./example_figures**: containes exemplary figures.


