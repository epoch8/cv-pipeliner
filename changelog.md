# 0.7.6

- Add callbacks arguments in Inferencers for progress bars

# 0.7.5

- Add parallel inference for object detection (need joblib)
- Classification Inferencer now accept ImageData generators

# 0.7.0+

- Apps moved to `epoch8/cv-demostand`
- Tensorflow is not required
- Reporter for detection/pipeline works faster

# 0.6.2 (2021-04-14)

- Improved algorithm for matching true and predicted images data (ImageDataMatching) 

# 0.6.0-0.6.1 (2021-04-01)

- Add Mean Expected Steps metrics
- Major apps code refactor (migrate from Streamlit to Dash)

# 0.5.1 (2021-03-12)

- Add presicion@K classification metrics
- Fix `cv_pipeliner.complex_pipelines` module bug

# 0.5.0 (2021-03-11)

- Detector now can be pipeline itself
- Add complex pipeline: TextDetection

# 0.4.5 (2021-01-18)

- Package `object_detection` is now not required

# 0.4.4 (2021-01-18)

- Annotation in dataset browser is made better: top-n is now showed in annotation list.
- Backups in annotation mode are now made every hour.

# 0.4.3 (2021-01-13)

- Added quick annotation mode in the main app for model quality research
