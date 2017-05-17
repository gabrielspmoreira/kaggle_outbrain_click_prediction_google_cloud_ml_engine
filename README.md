## A Google Cloud ML Engine PoC for Kaggle's Outbrain Click Prediction competition

I've jumped into [Outbrain Click Prediction](https://www.kaggle.com/c/outbrain-click-prediction) competition on Kaggle. After more than three months hicking to the top, I ended up in the 19th position (top 2%).  
I've published that journey in this [post series](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d), explaining how I got such a result, mainly due to Feature Engineering techniques and Google Cloud Platform.

After the competition, I tried to explore [Tensorflow](http://tensorflow.org/) and [Google Cloud Machine Learning Engine](https://cloud.google.com/products/machine-learning/) to see the results I could get with such tech stack.

The main motivation was a Google Research promissing technique named [Wide & Deep Learning](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html), which is very suitable for problems with sparse inputs (categorical features with large number of possible values), like click prediction and fraud detection. It trains together a scalable linear model (for memorization) and deep neural network (for generalization). This PoC was based on a [Google example](https://cloud.google.com/blog/big-data/2017/02/using-google-cloud-machine-learning-to-predict-clicks-at-scale) of this technique on Google Cloud ML Engine, using [Tensorflow](http://tensorflow.org/) and [Dataflow](https://cloud.google.com/dataflow/).

This PoC consists of three parts:

### Data Munging and Feature Engineering on Dataproc (Spark)

As first step, we used Spark SQL deployed on [Google Dataproc](https://cloud.google.com/dataproc/) to join the large relational database provided by Outbrain, engineer new features, split train and validation sets, and export to CSV files stored on [Google Cloud Storage](https://cloud.google.com/storage/), as described in this [post series](https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d). The Jupyter notebook used for this task is available in preprocessing/Outbrain-ProcessFeatureVectors.ipynb.

### Data Transformation on Dataflow (Apache Beam)

The second step was an Apache Beam script run on [Google Dataflow](https://cloud.google.com/dataflow/) to transform input CSVs datasets into Tensorflow [Example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) format. Some feature engineering techniques, like binning and log transformation of numerical features was also performed at this step, using [tf.Transform](https://research.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html).

### Model Training and Evaluation on Machine Learning Engine (Tensorflow)

The third step was to set up a Wide & Deep model using Tensorflow, using a [DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier), and deploy on Google Machine Learning Engine. 

This process uses pre-processed TFRecords data of the previous step. The features of the Linear and DNN models are defined separately, as can be seen in get_feature_columns() function from trainer/task.py. Linear model features include some binned numeric features and two-paired interactions among large categorical features, hashed to limit the feature space. Deep model features include scaled numerical features, and for categorical features we use an [embedding column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embedding_column).

#TODO  

* Add script to setup Dataproc cluster with Jupyter
* Add Users profile notebook to the repo, exporting on matching folders
* Add ALS Collaborative Filtering notebook
* Improve readme description and explain about implemented MAP evaluation


## Prerequisites
*   Upload Outbrain competition files to a bucket on GCS
*   Deploy a Google Dataproc cluster with the sample script on preprocessing folder, which [installs a Jupyter Notebook](https://cloud.google.com/dataproc/docs/tutorials/jupyter-notebook) on master node.
*   Upload Outbrain-ProcessFeatureVectors.ipynb to the Jupyter server running on your Dataproc cluster master. Change the input and output buckets in the notebook and run to perform the pre-processing and export the CSV files.
*   Make sure you follow the Google Cloud ML setup
    [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up) before
    trying the sample. More documentation about Cloud ML is available
    [here](https://cloud.google.com/ml/docs/).
*   Make sure you have installed
    [Tensorflow](https://www.tensorflow.org/install/) if you want to run the
    sample locally.
*   Make sure you have installed
    [tensorflow-transform](https://github.com/tensorflow/transform).
*   Make sure your Google Cloud project has sufficient quota.
*   Tensorflow has a dependency on a particular version of protobuf. Please run
    the following after installing tensorflow-transform:
```
pip install --upgrade protobuf==3.1.0
```

## Data Munging and Feature Engineering on Dataproc


## Data Transformation on Dataflow

The pre-processing step can be performed either locally or on cloud depending
upon the size of input data.

### Local Run

We recommend using local preprocessing only for testing on a small subset of the
data. You can run it as:

```
LOCAL_DATA_DIR=[download location]
head -7000 $LOCAL_DATA_DIR/train_feature_vectors_integral_eval.csv > $LOCAL_DATA_DIR/train-7k.txt
tail -3000 $LOCAL_DATA_DIR/train_feature_vectors_integral_eval.csv > $LOCAL_DATA_DIR/eval-3k.txt
LOCAL_DATA_PREPROC_DIR=$LOCAL_DATA_DIR/preproc_10k
python dataflow_preprocess.py --training_data $LOCAL_DATA_DIR/train-7k.txt \
                     --eval_data $LOCAL_DATA_DIR/eval-3k.txt \
                     --output_dir $LOCAL_DATA_PREPROC_DIR                     
```

### Cloud Run

In order to run pre-processing on the Cloud run the commands below.

```
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET="gs://${PROJECT}-ml"


PROJECT=ciandt-cognitive-sandbox
GCS_BUCKET="gs://${PROJECT}-ml"
GCS_PATH=${GCS_BUCKET}/outbrain/wide_n_deep
GCS_TRAIN_CSV=gs://ciandt-cognitive-kaggle/outbrain-click-prediction/tmp/train_feature_vectors_integral_eval.csv/part-*
GCS_VALIDATION_CSV=gs://ciandt-cognitive-kaggle/outbrain-click-prediction/tmp/validation_feature_vectors_integral.csv/part-*

python dataflow_preprocess.py --training_data $GCS_TRAIN_CSV \
                     --eval_data $GCS_VALIDATION_CSV \
                     --output_dir $GCS_PATH/tfrecords_preproc_with_bins4_no_shuffle \
                     --project_id $PROJECT \
                     --cloud
```

## Models

The sample implements a linear model trained with SDCA, as well a deep neural
network model. The code can be run either locally or on cloud.

### Local Run For the Small Dataset

Run the code as below:

#### Help options

```
  python -m trainer.task -h
```

#### How to run code

To train the linear model:

```
python -m trainer.task \
          --dataset kaggle \
          --l2_regularization 60 \
          --train_data_paths $LOCAL_OUTPUT_DIR/features_train* \
          --eval_data_paths $LOCAL_OUTPUT_DIR/features_eval* \
          --raw_metadata_path $LOCAL_OUTPUT_DIR/raw_metadata \
          --transformed_metadata_path $LOCAL_OUTPUT_DIR/transformed_metadata \
          --transform_savedmodel $LOCAL_OUTPUT_DIR/transform_fn \
          --output_path $TRAINING_OUTPUT_PATH
```

To train the deep model:

```
python -m trainer.task \
          --dataset kaggle \
          --model_type deep \
          --hidden_units 600 600 600 600 \
          --batch_size 512 \
          --train_data_paths $LOCAL_OUTPUT_DIR/features_train* \
          --eval_data_paths $LOCAL_OUTPUT_DIR/features_eval* \
          --raw_metadata_path $LOCAL_OUTPUT_DIR/raw_metadata \
          --transformed_metadata_path $LOCAL_OUTPUT_DIR/transformed_metadata \
          --transform_savedmodel $LOCAL_OUTPUT_DIR/transform_fn \
          --output_path $TRAINING_OUTPUT_PATH
```

Running time varies depending on your machine. Typically the linear model takes
at least 2 hours to train, and the deep model more than 8 hours. You can use
[Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/) to
follow the job's progress.

### Cloud Run 

You can train using either a single worker or using
multiple workers and parameter servers (ml-engine-config-small.yaml).

To train the linear model:

```
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
  --batch_size 256 \
  --train_set_size 55000000 \
  --eval_set_size 27380257 \
  --num_epochs 5 \
  --eval_steps 30 \
  --output_path "${GCS_OUTPUT_DIR}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_DIR}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_DIR}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_DIR}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_DIR}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_DIR}/features_train*"
```

To train the linear model without crosses, add the option `--ignore_crosses`

To train the deep model:

```
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
  --batch_size 256 \
  --hidden_units 1024 1024 1024 1024 1024 1024 \
  --train_set_size 55000000 \
  --eval_set_size 27380257 \
  --num_epochs 5 \
  --eval_steps 30 \
  --output_path "${GCS_OUTPUT_DIR}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_DIR}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_DIR}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_DIR}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_DIR}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_DIR}/features_train*"
```

To train Wide & Deep model:

To train the deep model:

```
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
  --batch_size 256 \
  --hidden_units 1024 1024 1024 1024 1024 1024 \
  --train_set_size 55000000 \
  --eval_set_size 27380257 \
  --num_epochs 5 \
  --eval_steps 30 \
  --output_path "${GCS_OUTPUT_DIR}/${JOB_ID}" \
  --raw_metadata_path "${GCS_PREPROC_DIR}/raw_metadata" \
  --transformed_metadata_path "${GCS_PREPROC_DIR}/transformed_metadata" \
  --transform_savedmodel "${GCS_PREPROC_DIR}/transform_fn" \
  --eval_data_paths "${GCS_PREPROC_DIR}/features_eval*" \
  --train_data_paths "${GCS_PREPROC_DIR}/features_train*"
  ```
When using the [distributed configuration](config-small.yaml), the linear model
may take as little as X minutes to train, and the deep model should finish in
around X minutes. Again you can point Tensorboard to the output path to follow
training progress.