## A Wide & Deep model deployed on Google Cloud ML Engine for Kaggle's Outbrain Click Prediction competition

I've jumped into [Outbrain Click Prediction](https://www.kaggle.com/c/outbrain-click-prediction) Kaggle competition by Oct. 2016. After more than three months climbing to the top, I ended up in the *19th position (top 2%)*.  
I've published that journey in this [post series](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d), explaining how I got such a result, mainly due to Feature Engineering techniques and Google Cloud Platform.

After the competition, I ran some experiments using [Tensorflow](http://tensorflow.org/), [Google Cloud Machine Learning Engine](https://cloud.google.com/products/machine-learning/) and [Dataflow](https://cloud.google.com/dataflow/) on competition data, to see the results I could get with such tech stack.

The main motivation was a promissing technique published by Google Research named [Wide & Deep Learning](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html), which is very suitable for problems with sparse inputs (categorical features with large number of possible values), like click prediction and fraud detection. It trains together a scalable linear model (for memorization) and a deep neural network (for generalization).  
This PoC was based on a [Google's example](https://cloud.google.com/blog/big-data/2017/02/using-google-cloud-machine-learning-to-predict-clicks-at-scale) of this technique using [Tensorflow](http://tensorflow.org/), [Google Cloud Machine Learning Engine](https://cloud.google.com/products/machine-learning/), and [Dataflow](https://cloud.google.com/dataflow/).

This PoC consists of a pipeline with three steps:

### 1 - Data Munging and Feature Engineering on Google Dataproc (Spark)

As first step, we used Spark SQL deployed on [Google Dataproc](https://cloud.google.com/dataproc/) to join the large relational database provided by Outbrain, engineer new features, and split train and validation sets, as described in this [post series](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d). Finally, the final DataFrame is exported to CSV files to [Google Cloud Storage](https://cloud.google.com/storage/), for usage in the next steps. The Jupyter notebooks used for this task are available under [preprocessing](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/spark/preprocessing/) folder .

### 2 - Data Transformation on Google Dataflow (Apache Beam)

The second step was an Apache Beam script run on [Google Dataflow](https://cloud.google.com/dataflow/) to transform the exported CSVs into Tensorflow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) format. Some feature engineering techniques, like binning and log transformation of numerical features was also performed at this step, using [tf.Transform](https://research.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html).

### 3 - Model Training and Evaluation on Google Machine Learning Engine (Tensorflow)

The third step was the set up of a [Wide & Deep Learning model](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) using Tensorflow ([DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier)), and deployment on Google Machine Learning Engine. 

That step used the pre-processed TFRecords data of the previous step. Features of the Linear and DNN models are defined separately, as can be seen in [`get_feature_columns()`](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/trainer/task.py#L140) function. Linear model features include some binned numeric features and two-paired interactions among large categorical features, hashed to limit the feature space. Deep model features include scaled numerical features, and for categorical features we use [embedding columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embedding_column).

For the evaluation, I've implemented the official metric for the competition Leaderboard: [MAP (Mean Average Precision)](https://www.kaggle.com/c/outbrain-click-prediction#evaluation). It was necessary to implement it as a custom MetricSpec to be used with the [Experiment](tf.contrib.learn.Experiment) class. I've also implemented a metric named "MAP_with_Leaked_Clicks", whose calculation is identical to MAP, but it takes into account leaked clicked ads (as described in this [post](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-ii-3645d92282b8)) and put them in the first position, because in the end any submission to Kaggle's leaderboard should consider the data leak for a better score.

## Prerequisites
* An account on [Google Cloud Platform](https://cloud.google.com/)
* Installation of [Google Cloud SDK](https://cloud.google.com/sdk/downloads) in the local computer
* A bucket on [Google Cloud Storage](https://cloud.google.com/storage/) with [Regional location](https://cloud.google.com/storage/docs/bucket-locations).
* Enable [Google Cloud Dataproc API](https://cloud.google.com/dataproc/docs/quickstarts/quickstart-gcloud),  Google Cloud Dataproc API,  Google Cloud Machine Learning Engine API on API Manager ([Google Cloud Console](https://console.cloud.google.com/apis/dashboard))
* Google Cloud ML setup
    [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up) before
    trying the sample. More documentation about Cloud ML is available
    [here](https://cloud.google.com/ml/docs/).
* Make sure your Google Cloud project has sufficient quota.    
* If you want to run the
    example locally for debugging, install [Tensorflow (tested with 1.0.0)](https://www.tensorflow.org/install/) and
    [tensorflow-transform](https://github.com/tensorflow/transform).

  * Tensorflow has a dependency on a particular version of protobuf. Please run
    the following after installing tensorflow-transform:
```
pip install --upgrade protobuf==3.1.0
```

# Running this PoC

## 1 - Data Munging and Feature Engineering on Dataproc

1. Upload all the [Outbrain competition CSV files](https://www.kaggle.com/c/outbrain-click-prediction) to a bucket on [Google Cloud Storage](https://cloud.google.com/storage/)
2. Start a Dataproc cluster with Jupyter setup on master node, running below commands. It will create a Dataproc cluster with 1 master node and 8 worker nodes, suitable to run the preprocessing Jupyter notebooks in about 10 hours. This startup script also installs a [Jupyter Notebook](https://cloud.google.com/dataproc/docs/tutorials/jupyter-notebook) on master node.
```bash
PROJECT_NAME="<project name>"
cd spark/scripts    
./dataproc_setup.sh $PROJECT_NAME outbrain-dataproc-cluster start
```
3. Open a web interface with Dataproc master node with the following command. It will open a Chrome browser window with the Jupyter notebook page.
```bash
./dataproc_setup.sh $PROJECT_NAME outbrain-dataproc-cluster browse
```
4. Use Jupyter Notebook page to upload files under [spark/preprocessing](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/tree/master/spark/preprocessing) folder to your Dataproc cluster master. 
5. Open the [1-Outbrain-ValidationSetSplit.ipynb](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/spark/preprocessing/1-Outbrain-ValidationSetSplit.ipynb) notebook.  
Set DATA_BUCKET_FOLDER variable with the GCS path for CSV data provided by competition (example below), and OUTPUT_BUCKET_FOLDER with an output GCS path. Run the notebook to generate a fixed validation set (for evaluation).
```python
evaluation = True
evaluation_verbose = False
OUTPUT_BUCKET_FOLDER = "gs://<GCS_BUCKET_NAME>/outbrain-click-prediction/output/"
DATA_BUCKET_FOLDER = "gs://<GCS_BUCKET_NAME>/outbrain-click-prediction/data/"
```
6. Open [2-Outbrain-UserProfiles.ipynb](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/spark/preprocessing/2-Outbrain-UserProfiles.ipynb), set the GCS paths like in the previous step and run the notebook. If evaluation == True, the output data will aggregate users info on a partial train set, ignoring visits in the validation set to avoid overfitting. Otherwise, the all visits in the train set will be used to build the user profiles. Then, run the notebook to output the aggregated users profiles.
```python
evaluation = True
evaluation_verbose = False
OUTPUT_BUCKET_FOLDER = "gs://<GCS_BUCKET_NAME>/outbrain-click-prediction/output/"
DATA_BUCKET_FOLDER = "gs://<GCS_BUCKET_NAME>/outbrain-click-prediction/data/"
```
7. Open [3-Outbrain-Preprocessing.ipynb](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/spark/preprocessing/3-Outbrain-Preprocessing.ipynb) notebook and set the GCS paths like in the previous step. Run the notebook to perform the pre-processing and export the merged CSV files with engineered features.  
If evaluation == True, the output CSVs will be a pre-processed partial train set and the validation set, otherwise, the output will be the pre-processed full train set and the test set (for which we are expected to run the predictions and submit to the competition). 
8. Stop Dataproc cluster with the following command:  
```bash
./dataproc_setup.sh $PROJECT_NAME outbrain-dataproc-cluster stop
```

## 2 - Data Transformation on Dataflow

The data transformation step can be performed either locally (on a data sample to debug) or on cloud (necessary to process the full data because of its size).

### Local Run

We recommend using local preprocessing only for testing on a small subset of the
data. You can run it as:

```bash
LOCAL_DATA_DIR=[download location]
head -7000 $LOCAL_DATA_DIR/train_feature_vectors_integral_eval_part_00001.csv > $LOCAL_DATA_DIR/train-7k.txt
tail -3000 $LOCAL_DATA_DIR/train_feature_vectors_integral_eval_part_00001.csv > $LOCAL_DATA_DIR/eval-3k.txt
LOCAL_DATA_PREPROC_DIR=$LOCAL_DATA_DIR/preproc_10k
python dataflow_preprocess.py --training_data $LOCAL_DATA_DIR/train-7k.txt \
                     --eval_data $LOCAL_DATA_DIR/eval-3k.txt \
                     --output_dir $LOCAL_DATA_PREPROC_DIR                     
```

### Cloud Run

In order to run pre-processing on the Cloud, run the commands below.

```bash
PROJECT="<project_name>"
GCS_BUCKET="gs://${PROJECT}-ml"
GCS_PATH=${GCS_BUCKET}/outbrain/wide_n_deep
GCS_PREPROC_PATH=$GCS_PATH/tfrecords_output
GCS_TRAIN_CSV=gs://{GCS_BUCKET}/outbrain/output/train_feature_vectors_integral_eval.csv/part-*
#Full training set (for submission)
#GCS_TRAIN_CSV=gs://{GCS_BUCKET}/outbrain/output/train_feature_vectors_integral.csv/part-*
GCS_VALIDATION_TEST_CSV=gs://{GCS_BUCKET}/outbrain-click-prediction/output/validation_feature_vectors_integral.csv/part-*
#Test set (for submission)
#GCS_VALIDATION_TEST_CSV=gs://{GCS_BUCKET}/outbrain-click-prediction/output/test_feature_vectors_integral.csv/part-*

python dataflow_preprocess.py --training_data $GCS_TRAIN_CSV \
                     --eval_data $GCS_VALIDATION_TEST_CSV \
                     --output_dir $GCS_PREPROC_PATH \
                     --project_id $PROJECT \
                     --cloud
```

## 3 - Model Training and Evaluation on Machine Learning Engine

This example implements a Wide & Deep model for click prediction. This step can be run either locally (to debug) or on cloud.

### Local Run For the Small Dataset

Run the code as below:

#### Help options

```
  python -m trainer.task -h
```

#### How to run code

To train the linear model:

```bash
python -m trainer.task \
      --model_type wide \
      --linear_l1_regularization 0.1 \
      --linear_l2_regularization 0.0 \
      --linear_learning_rate 0.05 \
      --train_set_size 7000 \
      --eval_set_size 3000 \
      --train_batch_size 256 \
      --num_epochs 5 \
      --ignore_crosses \
      --train_data_paths "$LOCAL_DATA_PREPROC_DIR/features_train*" \
      --eval_data_paths "$LOCAL_DATA_PREPROC_DIR/features_eval*" \
      --raw_metadata_path $LOCAL_DATA_PREPROC_DIR/raw_metadata \
      --transformed_metadata_path $LOCAL_DATA_PREPROC_DIR/transformed_metadata \
      --transform_savedmodel $LOCAL_DATA_PREPROC_DIR/transform_fn \
      --output_path $TRAINING_OUTPUT_PATH
```

To train the deep model:

```bash
python -m trainer.task \      
      --model_type deep \
      --deep_l1_regularization 1 \
      --deep_l2_regularization 1 \
      --hidden_units 512 256 128 \
      --train_batch_size 256 \
      --train_set_size 7000 \
      --eval_set_size 3000 \      
      --num_epochs 5 \
      --train_data_paths "$LOCAL_DATA_PREPROC_DIR/features_train*" \
      --eval_data_paths "$LOCAL_DATA_PREPROC_DIR/features_eval*" \
      --raw_metadata_path $LOCAL_DATA_PREPROC_DIR/raw_metadata \
      --transformed_metadata_path $LOCAL_DATA_PREPROC_DIR/transformed_metadata \
      --transform_savedmodel $LOCAL_DATA_PREPROC_DIR/transform_fn \
      --output_path $TRAINING_OUTPUT_PATH          
```

Running time varies depending on your machine. You can use
[Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) to
follow the job's progress.

### Cloud Run 

You can train using either a single worker or using
multiple workers and parameter servers (ml-engine-config-small.yaml).

Set environment variables:

```bash
GCS_MODEL_OUTPUT_PATH=$GCS_PATH/model_output
```


To train the linear model:

```bash
JOB_ID="out_wide_n_deep_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$GCS_BUCKET" \
  --region us-east1 \
  --config ml-engine-config-small.yaml \
  --runtime-version 1.0 \
  --async \
  -- \
  --model_type wide \
  --linear_l1_regularization 0.03 \
  --linear_l2_regularization 100 \
  --linear_learning_rate 0.01 \
  --train_batch_size 128 \
  --num_epochs 3 \
  --train_set_size 55000000 \
  --eval_set_size 27380257 \
  --full_evaluation_after_training \
  --output_path "${GCS_OUTPUT_DIR}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_DIR}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_DIR}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_DIR}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_DIR}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_DIR}/features_train*"
```

To train the linear model without feature interactions, add the option `--ignore_crosses`

To train the deep model:

```bash
JOB_ID="out_wide_n_deep_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$GCS_BUCKET" \
  --region us-east1 \
  --config ml-engine-config-small.yaml \
  --runtime-version 1.0 \
  --async \
  -- \
  --model_type deep \
  --deep_l1_regularization 0.0 \
  --deep_l2_regularization 0.001 \
  --deep_learning_rate 0.05 \
  --deep_dropout 0.15 \
  --deep_hidden_units "1024 1024 1024 1024 1024" \
  --deep_embedding_size_factor 8 \
  --train_batch_size 128 \
  --num_epochs 3 \
  --train_set_size 55000000 \
  --eval_set_size 27380257 \
  --full_evaluation_after_training \
  --output_path "${GCS_OUTPUT_DIR}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_DIR}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_DIR}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_DIR}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_DIR}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_DIR}/features_train*"
```

To train Wide & Deep model:


```bash
JOB_ID="out_wide_n_deep_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$GCS_BUCKET" \
  --region us-east1 \
  --config ml-engine-config-small.yaml \
  --runtime-version 1.0 \
  --async \
  -- \
  --model_type wide_n_deep \
  --linear_l1_regularization 0.03 \
  --linear_l2_regularization 100 \
  --linear_learning_rate 0.01 \
  --deep_l1_regularization 0.0 \
  --deep_l2_regularization 0.001 \
  --deep_learning_rate 0.05 \
  --deep_dropout 0.15 \
  --deep_hidden_units "1024 1024 1024 1024 1024" \
  --deep_embedding_size_factor 8 \
  --train_batch_size 128 \
  --num_epochs 3 \
  --train_set_size 55000000 \
  --eval_set_size 27380257 \
  --full_evaluation_after_training \
  --output_path "${GCS_OUTPUT_DIR}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_DIR}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_DIR}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_DIR}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_DIR}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_DIR}/features_train*"
  ```

The default behaviour is that validation set is evaluated on batches as training evolves (for learning curve assessment). If you want to perform the evaluation of the full validation set (~27 Million samples) only after the training was completed (--full_evaluation_after_training), the final "MAP_with_Leaked_Clicks" will be a credible proxy for the submission score on the Leaderboard (for a model trained with the full train set to predict clicks on test set).

Using the [config file](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/ml-engine-config-small.yaml) provided as example, the linear model
may take as little as 5h30m to train, and the deep model should finish in
around 12h, and Wide & Deep model takes about 9h30m. You can run [Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) and monitor training progress, using the following command:

```bash
tensorboard --logdir ${GCS_OUTPUT_DIR}/${JOB_ID}
```


### HyperTune

Google Cloud Machine Learning Engine features a hyperparameter tuning engine named [HyperTune](https://cloud.google.com/ml-engine/docs/how-tos/using-hyperparameter-tuning). It allows you to define which hyperparameters you want to tune, what are their ranges and scale (numeric params) or possible values (categorical / discrete params). According to Google documentation, HyperTune uses a smart strategy to balance exploration of the hyperparameters search space, and exploitation of the values that resulted in more accurate models.

The [ml-engine-config-hypertune.yaml](https://github.com/gabrielspmoreira/kaggle_outbrain_click_prediction_google_cloud_ml_engine/blob/master/ml-engine-config-hypertune.yaml) config file is an example of how to setup a Hyperparameter Tuning job on Cloud ML. The file contains settings to control how many trials will be attempted, which hyperparameters will be tuned and the possible values.  
Thus, the of goal this tuning was to MAXIMIZE the metric "MAP_with_Leaked_Clicks", with a maximum of 30 trials, using a sample of 1M rows for training and 100K for evaluation (and tuning optimization). 

```bash
JOB_ID="out_wide_n_deep_${USER}_$(date +%Y%m%d_%H%M%S)"
gcloud ml-engine jobs submit training "$JOB_ID" \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$GCS_BUCKET" \
  --region us-east1 \
  --config ml-engine-config-hypertune.yaml  \
  --runtime-version 1.0 \
  --async \
  -- \
  --model_type wide_n_deep \
  --train_set_size 1000000 \
  --eval_set_size 100000 \
  --num_epochs 1 \
  --full_evaluation_after_training \
  --output_path "${GCS_MODEL_OUTPUT_PATH}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_PATH}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_PATH}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_PATH}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_PATH}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_PATH}/features_train*"
```

This HyperTune job should take about 8h to run the 30 trials with this cluster configuration. As it progress, the job outputs a JSON like below, with the trials sorted by models accuracy, and respective hyperparameters values.

```json
{
  "completedTrialCount": "30",
  "trials": [
    {
      "trialId": "6",
      "hyperparameters": {
        "train_batch_size": "256",
        "linear_learning_rate": "0.05",
        "deep_hidden_units": "1024 1024 1024",
        "deep_l2_regularization": "0.03",
        "deep_embedding_size_factor": "2",
        "deep_dropout": "0",
        "deep_l1_regularization": "0.01",
        "linear_l2_regularization": "1",
        "linear_l1_regularization": "0.01",
        "deep_learning_rate": "0.1"
      },
      "finalMetric": {
        "trainingStep": "618",
        "objectiveValue": 0.666045427322
      }
    },
    {
      "trialId": "21",
      "hyperparameters": {
        "train_batch_size": "128",
        "linear_learning_rate": "0.01",
        "deep_l2_regularization": "0.01",
        "deep_hidden_units": "1024 1024 1024",
        "deep_embedding_size_factor": "10",
        "deep_dropout": "0.1",
        "deep_l1_regularization": "0.3",
        "linear_l2_regularization": "30",
        "linear_l1_regularization": "0.03",
        "deep_learning_rate": "0.2"
      },
      "finalMetric": {
        "trainingStep": "662",
        "objectiveValue": 0.666007578373
      }
    },
    {
      "trialId": "8",
      "hyperparameters": {
        "deep_dropout": "0",
        "deep_embedding_size_factor": "4",
        "deep_l1_regularization": "0.01",
        "linear_l2_regularization": "0.1",
        "linear_l1_regularization": "0.03",
        "deep_learning_rate": "0.2",
        "train_batch_size": "256",
        "linear_learning_rate": "0.05",
        "deep_hidden_units": "1024 1024 1024",
        "deep_l2_regularization": "0.03"
      },
      "finalMetric": {
        "trainingStep": "641",
        "objectiveValue": 0.664392352104
      }
    },
    ...

  }
``` 
