# PolyDeep project
This project is part of the PolyDeep research project. See more details at http://polydeep.org

# Compi pipelines

Four different Compi pipelines were developed for the object-detection model development and evaluation: 
- (i) dataset split for development (training and validation datasets) and testing
- (ii) model development
- (iii) model evaluation (with images)
- (iv) video annotation. 

## Dataset split pipeline (`ttv.xml`)

This pipeline is a preamble pipeline containing the steps necessary to prepare the dataset for the subsequent model development and evaluation. This pipeline starts with the check-dataset step, which is a control step that checks whether the dataset has been downloaded locally and, if not, alerts the user and ends the execution. Then, the create-ttv step analyses the dataset to generate training, validation, and test subsets based on the number of images of each polyp (see Sect. 4.1). As a result, this step generates three files with the polyp identifiers that correspond to each subset of the data.

## Model development pipeline (`train.xml`)

This pipeline is responsible for the model development. This pipeline also starts with the check-dataset step, which is followed by another control step (check-gpu) that checks the sanity of the GPU memory, as a failure in it could cause erroneous results. After that, the model-development step performs the model training and validation, generating several files with the performance results. This step is followed by the generate-plot-data that post-processes previous result files in order to plot them in the three following steps: plot-loss, plot-map, and plot-metrics, which generate several plots that summarize the model evaluation results. Finally, a cleanup step removes some auxiliary files generated during the pipeline execution.

## Model evaluation pipeline (`test.xml`) 

This pipeline is responsible for evaluating the model selected in the development phase with a new/unseen dataset of images. This pipeline has the same control steps as the model development pipeline (i.e. check-dataset and check-gpu) and a new control step (check-neural-network) to check that the trained model exists. Then, the test step evaluates the model with the test dataset created with the dataset split pipeline, generating a file with the performance results.

## Video annotation pipeline (`video-annotation.xml`)

This pipeline allows applying the model to video segments of polyp and normal-mucosa regions to obtain annotated videos. This pipeline starts with the same control steps that the model evaluation pipeline. Then, this pipeline comprises three main stages: (i) download videos (download-polyp-videos and download-normal-mucosa-videos), (ii) extract video segments (extract-polyp-segments and extract-normal-mucosa-segments), and (iii) use the model to detect polyps in each frame of the video segments (predict-polyp-segments and predict-normal-mucosa-segments). A final a cleanup step removes some auxiliary files generated during the pipeline execution.
