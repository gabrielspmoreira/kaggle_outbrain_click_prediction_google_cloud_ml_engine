# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample for Criteo dataset can be run as a wide or deep model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math
import os
import sys

import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn import learn_runner, metric_spec
from tensorflow.python.ops import metrics
from tensorflow.contrib import metrics as metrics_lib

from tensorflow.python.platform import tf_logging as logging
tf.logging.set_verbosity(tf.logging.INFO)

from trainer.features import LABEL_COLUMN, DISPLAY_ID_COLUMN, DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN, CATEGORICAL_COLUMNS, DOC_CATEGORICAL_MULTIVALUED_COLUMNS, BOOL_COLUMNS, INT_COLUMNS, FLOAT_COLUMNS_LOG_BIN_TRANSFORM, FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM

from itertools import combinations

'''
tf.logging.set_verbosity(tf.logging.ERROR)

def set_verbose_logging(verbose_logging=False):
  if verbose_logging is True:
    tf.logging.set_verbosity(tf.logging.INFO)
'''

HASH_KEY = 42

MODEL_TYPES = ['wide', 'deep', 'wide_n_deep']
WIDE, DEEP, WIDE_N_DEEP = MODEL_TYPES

CROSSES = 'crosses'
NUM_EXAMPLES = 'num_examples'
L1_REGULARIZATION = 'l1_regularization'
L2_REGULARIZATION = 'l2_regularization'

KEY_FEATURE_COLUMN = 'example_id'
TARGET_FEATURE_COLUMN = 'label'

CROSS_HASH_BUCKET_SIZE = int(1e6)

MODEL_DIR = 'model'

FORMAT_CATEGORICAL_FEATURE_ID = 'categorical-feature-{}_id'
FORMAT_INT_FEATURE = 'int-feature-{}'

EVAL_DATASET_SIZE = 27380257
EVAL_BATCH_SIZE=3000

PIPELINE_CONFIG = {L1_REGULARIZATION: 10.0,
                   L2_REGULARIZATION: 1.0, 
                   NUM_EXAMPLES: 55000000}
#
# Pipeline config for the two datasets
# The data in the CROSSES is 1-based indexing
#
'''
PIPELINE_CONFIG = {
    KAGGLE: {
        NUM_EXAMPLES:
            45 * 1e6,
        L2_REGULARIZATION:
            60,
        CROSSES: [(27, 31), (33, 37), (27, 29), (4, 6), (19, 36), (19, 22),
                  (19, 33), (6, 9), (10, 5), (19, 35, 36), (30, 36), (30, 11),
                  (20, 30), (19, 22, 28), (27, 31, 39), (1, 8), (11, 5),
                  (11, 7), (25, 2), (26, 27, 31), (38, 5), (19, 22, 11),
                  (37, 5), (24, 11), (13, 4), (19, 8), (27, 31, 33),
                  (17, 19, 36), (31, 3), (26, 5), (30, 12), (27, 31, 2),
                  (11, 9), (15, 34), (19, 26, 36), (27, 36), (30, 5), (23, 37),
                  (13, 3), (31, 6), (26, 8), (30, 33), (27, 36, 37), (1, 6),
                  (17, 30), (20, 23), (27, 31, 35), (26, 1), (26, 27, 36)]
    },
    LARGE: {
        NUM_EXAMPLES:
            4 * 1e9,
        L2_REGULARIZATION:
            500,
        CROSSES: [(19, 12), (10, 12), (10, 11), (32, 12), (30, 1), (36, 39),
                  (13, 3), (26, 32), (15, 23), (10, 9), (20, 25), (16, 26, 32),
                  (11, 12), (30, 10), (15, 38), (10, 6), (39, 8), (39, 10),
                  (19, 28, 12), (15, 37), (26, 7), (11, 5), (14, 39, 8),
                  (11, 2), (12, 4), (28, 1), (26, 32, 11), (26, 10, 7),
                  (22, 30), (15, 24, 38), (20, 10, 12), (32, 9), (15, 8),
                  (32, 4), (26, 3), (29, 30), (22, 30, 39), (22, 30, 36, 39),
                  (22, 26), (20, 11), (4, 9), (26, 12), (12, 13), (32, 6),
                  (39, 11), (15, 26, 32)]
    }
}
'''

def create_parser():
  """Initialize command line parser using arparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  '''
  parser.add_argument(
      '--dataset',
      help='Criteo dataset to run training on',
      choices=DATASETS,
      required=True)
  '''
  parser.add_argument(
      '--model_type',
      help='Model type to train on',
      choices=MODEL_TYPES,
      default=WIDE)
  parser.add_argument(
      '--train_data_paths', type=str, action='append', required=True)
  parser.add_argument(
      '--eval_data_paths', type=str, action='append', required=True)
  parser.add_argument('--output_path', type=str, required=True)
  # The following three parameters are required for tf.Transform.
  parser.add_argument('--raw_metadata_path', type=str, required=True)
  parser.add_argument('--transformed_metadata_path', type=str, required=True)
  parser.add_argument('--transform_savedmodel', type=str, required=True)
  parser.add_argument(
      '--hidden_units',
      nargs='*',
      help='List of hidden units per layer. All layers are fully connected. Ex.'
      '`64 32` means first layer has 64 nodes and second one has 32.',
      default=[100, 50],
      type=int)
  parser.add_argument(
      '--batch_size',
      help='Number of input records used per batch',
      default=512,
      type=int)
  parser.add_argument(
      '--eval_batch_size',
      help='Number of eval records used per batch',
      default=EVAL_BATCH_SIZE,
      type=int)
  parser.add_argument(
      '--train_steps', help='Number of training steps to perform.', type=int)
  parser.add_argument(
      '--eval_steps',
      help='Number of evaluation steps to perform.',
      type=int)
  parser.add_argument(
      '--train_set_size',
      help='Number of samples on the train dataset.',
      type=int)
  parser.add_argument(
      '--eval_set_size',
      help='Number of samples on the train dataset.',
      type=int)
  parser.add_argument('--l1_regularization', help='L2 Regularization', type=int)
  parser.add_argument('--l2_regularization', help='L2 Regularization', type=int)
  parser.add_argument(
      '--num_epochs', help='Number of epochs', default=5, type=int)
  parser.add_argument(
      '--ignore_crosses',
      action='store_true',
      default=False,
      help='Whether to ignore crosses (linear model only).')
  return parser


def get_embedding_size(unique_val_count):
  return int(math.floor(6 * unique_val_count**0.25))

def get_feature_columns(config, model_type, vocab_sizes, use_crosses):
  """Return the feature columns with their names and types."""
  result = []
  boundaries = [1.5**j - 0.51 for j in range(40)]

  # TODO(b/35300113): Reduce the range and other duplication between this and
  # preprocessing.

  # TODO(b/35300113): Can iterate over metadata so that we don't need to
  # re-define the schema here?

  '''
  for index in range(1, 14):
    column = tf.contrib.layers.bucketized_column(
        tf.contrib.layers.real_valued_column(
            FORMAT_INT_FEATURE.format(index),
            dtype=tf.int64),
        boundaries)
    result.append(column)

  if model_type == LINEAR:
    for index in range(14, 40):
      column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      result.append(column)
    if use_crosses:
      for cross in config[CROSSES]:
        column = tf.contrib.layers.crossed_column(
            [result[index - 1] for index in cross],
            hash_bucket_size=CROSS_HASH_BUCKET_SIZE,
            hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
            combiner='sum')
        result.append(column)
  elif model_type == DEEP:
    for index in range(14, 40):
      column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
      vocab_size = vocab_sizes[column_name]
      column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name, vocab_size, combiner='sum')
      embedding_size = int(math.floor(6 * vocab_size**0.25))
      embedding = tf.contrib.layers.embedding_column(column,
                                                     embedding_size,
                                                     combiner='mean') 
      result.append(column)
  '''

  wide_columns = []
  deep_columns = []

  event_weekend = tf.contrib.layers.sparse_column_with_integerized_feature('event_weekend', bucket_size=2, dtype=tf.int16, combiner="sum") #0-1
  user_has_already_viewed_doc = tf.contrib.layers.sparse_column_with_integerized_feature("user_has_already_viewed_doc", bucket_size=2, dtype=tf.int16, combiner="sum") #0-1
  event_hour = tf.contrib.layers.sparse_column_with_integerized_feature("event_hour", bucket_size=7, dtype=tf.int64, combiner="sum") #1-6
  event_platform = tf.contrib.layers.sparse_column_with_integerized_feature("event_platform", bucket_size=4, dtype=tf.int64, combiner="sum") #1-3
  traffic_source = tf.contrib.layers.sparse_column_with_integerized_feature("traffic_source", bucket_size=4, dtype=tf.int64, combiner="sum") #1-3

  if 'wide' in model_type:
    '''
    event_weekend = tf.contrib.layers.sparse_column_with_hash_bucket('event_weekend', hash_bucket_size=4, dtype=tf.int16, combiner="sum") #0-1
    user_has_already_viewed_doc = tf.contrib.layers.sparse_column_with_hash_bucket("user_has_already_viewed_doc", hash_bucket_size=4, dtype=tf.int16, combiner="sum") #0-1
    event_hour = tf.contrib.layers.sparse_column_with_hash_bucket("event_hour", hash_bucket_size=12, dtype=tf.int64, combiner="sum") #1-6
    event_platform = tf.contrib.layers.sparse_column_with_hash_bucket("event_platform", hash_bucket_size=6, dtype=tf.int64, combiner="sum") #1-3
    traffic_source = tf.contrib.layers.sparse_column_with_hash_bucket("traffic_source", hash_bucket_size=6, dtype=tf.int64, combiner="sum") #1-3
    '''    

    #Single-valued categories
    ad_id = tf.contrib.layers.sparse_column_with_hash_bucket('ad_id', hash_bucket_size=250000, dtype=tf.int64, combiner="sum")   #418295   
    doc_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_id', hash_bucket_size=100000, dtype=tf.int64, combiner="sum") #143856
    doc_event_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_event_id', hash_bucket_size=300000, dtype=tf.int64, combiner="sum") #636482
    ad_advertiser = tf.contrib.layers.sparse_column_with_hash_bucket('ad_advertiser', hash_bucket_size=2500, dtype=tf.int32, combiner="sum") #2052
    doc_ad_publisher_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_ad_publisher_id', hash_bucket_size=1000, dtype=tf.int32, combiner="sum") #830
    doc_ad_source_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_ad_source_id', hash_bucket_size=4000, dtype=tf.int32, combiner="sum") #6339
    doc_event_publisher_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_event_publisher_id', hash_bucket_size=1000, dtype=tf.int32, combiner="sum") #830
    doc_event_source_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_event_source_id', hash_bucket_size=4000, dtype=tf.int32, combiner="sum") #6339
    event_country = tf.contrib.layers.sparse_column_with_hash_bucket('event_country', hash_bucket_size=300, dtype=tf.int32, combiner="sum")  #222
    event_country_state = tf.contrib.layers.sparse_column_with_hash_bucket('event_country_state', hash_bucket_size=2000, dtype=tf.int32, combiner="sum") #1892
    event_geo_location = tf.contrib.layers.sparse_column_with_hash_bucket('event_geo_location', hash_bucket_size=2500, dtype=tf.int32, combiner="sum") #2273
    
    #Multi-valued categories
    doc_ad_category_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_ad_category_id', hash_bucket_size=100, dtype=tf.int32, combiner="sum") #90
    doc_ad_topic_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_ad_topic_id', hash_bucket_size=350, dtype=tf.int32, combiner="sum") #301
    doc_ad_entity_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_ad_entity_id', hash_bucket_size=10000, dtype=tf.int64, combiner="sum") #52439
    doc_event_category_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_event_category_id', hash_bucket_size=100, dtype=tf.int32, combiner="sum") #90
    doc_event_topic_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_event_topic_id', hash_bucket_size=350, dtype=tf.int32, combiner="sum") #301
    doc_event_entity_id = tf.contrib.layers.sparse_column_with_hash_bucket('doc_event_entity_id', hash_bucket_size=10000, dtype=tf.int64, combiner="sum") #52439

    float_simple_binned = []
    for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
      field_name = name 
      float_simple_binned.append((field_name, tf.contrib.layers.sparse_column_with_integerized_feature(field_name+'_binned', bucket_size=15, dtype=tf.int16, combiner="sum")))
    float_simple_binned_dict = dict(float_simple_binned)

    float_log_binned = []
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
      field_name = name
      float_log_binned.append((field_name, tf.contrib.layers.sparse_column_with_integerized_feature(field_name+'_log_binned', bucket_size=15, dtype=tf.int16, combiner="sum")))
    float_log_binned_dict = dict(float_log_binned)

    int_log_binned = []
    for name in INT_COLUMNS:  
      field_name = name
      int_log_binned.append((field_name, tf.contrib.layers.sparse_column_with_integerized_feature(field_name+'_log_int', bucket_size=15, dtype=tf.int16, combiner="sum")))
    int_log_binned_dict = dict(int_log_binned)

    # Continuous columns can be converted to categorical via bucketization
    #age_buckets = layers.bucketized_column(
    #    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns 
    wide_columns = [event_weekend, user_has_already_viewed_doc, event_hour, event_platform, traffic_source,
                    ad_id, doc_id, doc_event_id, ad_advertiser, doc_ad_source_id, doc_event_publisher_id,
                    doc_event_source_id, event_country, event_country_state, event_geo_location,
                    doc_ad_category_id, doc_ad_topic_id, doc_ad_entity_id, 
                    doc_event_category_id, doc_event_topic_id, doc_event_entity_id
                    ] + float_simple_binned_dict.values() \
                      + float_log_binned_dict.values() \
                      + int_log_binned_dict.values()

    if use_crosses:
      wide_interaction_features = [
                                 ad_id, doc_id, doc_event_id, ad_advertiser, doc_ad_source_id,
                                 doc_ad_publisher_id, doc_event_publisher_id, doc_event_source_id, 
                                 event_country, event_country_state, event_geo_location,
                                 doc_ad_category_id, doc_ad_topic_id, doc_ad_entity_id,
                                 doc_event_category_id, doc_event_topic_id, doc_event_entity_id,
                                 event_weekend, user_has_already_viewed_doc, event_hour, event_platform, traffic_source] 
                                #+ [float_log_binned_dict[f] for f in [
                                #             'pop_ad_id',
                                #             'pop_ad_id_conf_multipl',
                                #             'pop_advertiser_id'
                                #             ]]

      full_interactions = combinations(wide_interaction_features, 2)

      #Combinations meaningless for prediction to ignore
      interactions_to_ignore = list(combinations([ad_id, ad_advertiser, doc_id, doc_ad_source_id, doc_ad_publisher_id] , 2)) + \
                               list(combinations([doc_event_id, doc_event_publisher_id, doc_event_source_id], 2)) + \
                               list(combinations([event_country, event_country_state, event_geo_location], 2)) + \
                               list(combinations([doc_event_category_id, doc_event_entity_id, doc_event_entity_id], 2)) + \
                               [(ad_id, doc_ad_category_id), 
                                (ad_id, doc_ad_topic_id), 
                                (ad_id, doc_ad_entity_id)] + \
                               [(doc_id, doc_ad_category_id), 
                                (doc_id, doc_ad_topic_id), 
                                (doc_id, doc_ad_entity_id)]
      '''
                               list(combinations([ad_id, doc_id, float_log_binned_dict['pop_ad_id']], 2)) + \
                               [(ad_id, float_log_binned_dict['pop_ad_id']),
                                (ad_id, float_log_binned_dict['pop_advertiser_id']),
                                (ad_id, float_log_binned_dict['pop_ad_id_conf_multipl']),
                                (doc_id, float_log_binned_dict['pop_ad_id']),
                                (doc_id, float_log_binned_dict['pop_advertiser_id']),
                                (doc_id, float_log_binned_dict['pop_ad_id_conf_multipl']),
                                (float_log_binned_dict['pop_ad_id'], float_log_binned_dict['pop_ad_id_conf_multipl']),
                                (float_log_binned_dict['pop_ad_id'], float_log_binned_dict['pop_advertiser_id']),
                                (float_log_binned_dict['pop_ad_id_conf_multipl'], float_log_binned_dict['pop_advertiser_id'])
                                ]
      '''

      meaningful_interactions = set(full_interactions) - set(interactions_to_ignore) - \
                                set(map(lambda x: (x[1], x[0]), interactions_to_ignore))

      for interaction in meaningful_interactions:       
        bucket_size = interaction[0].bucket_size * interaction[1].bucket_size
        #bucket_size = int((interaction[0].bucket_size * interaction[1].bucket_size) / 1000.0)
        #If both categorical features are sparse, reduce their space to something manageable
        if not (interaction[0].is_integerized and interaction[1].is_integerized):
          bucket_size = int(math.pow(bucket_size, 0.78))
        wide_columns.append(tf.contrib.layers.crossed_column(interaction, hash_key=HASH_KEY, combiner="sum",
                               hash_bucket_size=bucket_size)
                            )

      '''
      wide_columns.extend([
                    #feature interactions
                    tf.contrib.layers.crossed_column([ad_id, doc_event_id], hash_key=HASH_KEY, combiner="sum",
                                     hash_bucket_size=int(1e7)),
                    tf.contrib.layers.crossed_column([ad_id, doc_event_publisher_id], hash_key=HASH_KEY, combiner="sum",
                                     hash_bucket_size=int(1e7)),
                    tf.contrib.layers.crossed_column([ad_id, doc_event_source_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e5)),
                    tf.contrib.layers.crossed_column([doc_id, doc_event_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e7)),
                    tf.contrib.layers.crossed_column([ad_advertiser, doc_event_publisher_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([ad_advertiser, doc_event_publisher_id, event_country], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([ad_advertiser, doc_event_source_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_source_id, doc_event_source_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([ad_id, event_country], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([ad_id, event_country_state], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([ad_advertiser, event_country], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e5)),
                    tf.contrib.layers.crossed_column([ad_advertiser, event_country_state], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e5)),
                    tf.contrib.layers.crossed_column([ad_advertiser, event_geo_location], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([doc_ad_category_id, doc_event_category_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_topic_id, doc_event_topic_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_entity_id, doc_event_entity_id], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e7)),
                    tf.contrib.layers.crossed_column([doc_ad_category_id, event_country], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_topic_id, event_country], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_entity_id, event_country], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([doc_ad_category_id, event_country_state], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_topic_id, event_country_state], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([doc_ad_entity_id, event_country_state], hash_key=HASH_KEY, combiner="sum",
                                                     hash_bucket_size=int(1e6))
                   ])
      '''

  if 'deep' in model_type:
    event_weekend_ohe = tf.contrib.layers.one_hot_column(event_weekend) #0-1
    user_has_already_viewed_doc_ohe = tf.contrib.layers.one_hot_column(user_has_already_viewed_doc) #0-1
    event_hour_ohe = tf.contrib.layers.one_hot_column(event_hour) #1-6
    event_platform_ohe = tf.contrib.layers.one_hot_column(event_platform) #1-3
    traffic_source_ohe = tf.contrib.layers.one_hot_column(traffic_source) #1-3
    
    # Continuous columns
    #continuous_columns = [tf.contrib.layers.real_valued_column(c) for c in FLOAT_COLUMNS + INT_COLUMNS]
    #continuous_columns = [tf.contrib.layers.real_valued_column(c) for c in FLOAT_COLUMNS]

    float_columns_simple = []
    for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
      float_columns_simple.append(tf.contrib.layers.real_valued_column(name))

    float_columns_log_01scaled = []
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
      float_columns_log_01scaled.append(tf.contrib.layers.real_valued_column(name+'_log_01scaled'))

    int_columns_log_01scaled = []
    for name in INT_COLUMNS:
      int_columns_log_01scaled.append(tf.contrib.layers.real_valued_column(name+'_log_01scaled'))

    '''
    deep_columns = [  event_weekend_ohe,
                      user_has_already_viewed_doc_ohe,
                      event_hour_ohe,
                      event_platform_ohe,
                      traffic_source_ohe,
                      #Single-valued categories
                      tf.contrib.layers.scattered_embedding_column('ad_id', size=250000*get_embedding_size(250000), dimension=get_embedding_size(250000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_id', size=100000*get_embedding_size(100000), dimension=get_embedding_size(100000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_id', size=300000*get_embedding_size(300000), dimension=get_embedding_size(300000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('ad_advertiser', size=2500*get_embedding_size(2500), dimension=get_embedding_size(2500), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_source_id', size=4000*get_embedding_size(4000), dimension=get_embedding_size(4000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_publisher_id', size=1000*get_embedding_size(1000), dimension=get_embedding_size(1000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_source_id', size=4000*get_embedding_size(4000), dimension=get_embedding_size(4000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('event_country', size=300*get_embedding_size(300), dimension=get_embedding_size(300), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('event_country_state', size=2000*get_embedding_size(2000), dimension=get_embedding_size(2000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('event_geo_location', size=2500*get_embedding_size(2500), dimension=get_embedding_size(2500), hash_key=HASH_KEY, combiner="mean"),
                      #Multi-valued categories
                      tf.contrib.layers.scattered_embedding_column('doc_ad_category_id', size=100*get_embedding_size(100), dimension=get_embedding_size(100), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_topic_id', size=350*get_embedding_size(350), dimension=get_embedding_size(350), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_entity_id', size=10000*get_embedding_size(10000), dimension=get_embedding_size(10000), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_category_id', size=100*get_embedding_size(100), dimension=get_embedding_size(100), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_topic_id', size=350*get_embedding_size(350), dimension=get_embedding_size(350), hash_key=HASH_KEY, combiner="mean"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_entity_id', size=10000*get_embedding_size(10000), dimension=get_embedding_size(10000), hash_key=HASH_KEY, combiner="mean"),
                    ]  + continuous_columns
    '''

    deep_columns = [ event_weekend_ohe,
                     user_has_already_viewed_doc_ohe,
                     event_hour_ohe,
                     event_platform_ohe,
                     traffic_source_ohe,
                      #Single-valued categories
                      tf.contrib.layers.scattered_embedding_column('ad_id', size=250000*get_embedding_size(250000), dimension=get_embedding_size(250000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_id', size=100000*get_embedding_size(100000), dimension=get_embedding_size(100000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_id', size=300000*get_embedding_size(300000), dimension=get_embedding_size(300000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('ad_advertiser', size=2500*get_embedding_size(2500), dimension=get_embedding_size(2500), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_publisher_id', size=1000*get_embedding_size(1000), dimension=get_embedding_size(4000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_source_id', size=4000*get_embedding_size(4000), dimension=get_embedding_size(4000), hash_key=HASH_KEY, combiner="sum"),                    
                      tf.contrib.layers.scattered_embedding_column('doc_event_publisher_id', size=1000*get_embedding_size(1000), dimension=get_embedding_size(1000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_source_id', size=4000*get_embedding_size(4000), dimension=get_embedding_size(4000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('event_country', size=300*get_embedding_size(300), dimension=get_embedding_size(300), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('event_country_state', size=2000*get_embedding_size(2000), dimension=get_embedding_size(2000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('event_geo_location', size=2500*get_embedding_size(2500), dimension=get_embedding_size(2500), hash_key=HASH_KEY, combiner="sum"),
                      #Multi-valued categories
                      tf.contrib.layers.scattered_embedding_column('doc_ad_category_id', size=100*get_embedding_size(100), dimension=get_embedding_size(100), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_topic_id', size=350*get_embedding_size(350), dimension=get_embedding_size(350), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_ad_entity_id', size=10000*get_embedding_size(10000), dimension=get_embedding_size(10000), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_category_id', size=100*get_embedding_size(100), dimension=get_embedding_size(100), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_topic_id', size=350*get_embedding_size(350), dimension=get_embedding_size(350), hash_key=HASH_KEY, combiner="sum"),
                      tf.contrib.layers.scattered_embedding_column('doc_event_entity_id', size=10000*get_embedding_size(10000), dimension=get_embedding_size(10000), hash_key=HASH_KEY, combiner="sum"),
                    ] + float_columns_simple + float_columns_log_01scaled + int_columns_log_01scaled    

  '''
  tf.contrib.layers.embedding_column(ad_id, dimension=get_embedding_size(ad_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_id, dimension=get_embedding_size(doc_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_event_id, dimension=get_embedding_size(doc_event_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(ad_advertiser, dimension=get_embedding_size(ad_advertiser.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_ad_source_id, dimension=get_embedding_size(doc_ad_source_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_event_publisher_id, dimension=get_embedding_size(doc_event_publisher_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_event_source_id, dimension=get_embedding_size(doc_event_source_id.length), combiner='mean'),     
  tf.contrib.layers.embedding_column(event_country, dimension=get_embedding_size(event_country.length), combiner='mean'),
  tf.contrib.layers.embedding_column(event_country_state, dimension=get_embedding_size(event_country_state.length), combiner='mean'),
  tf.contrib.layers.embedding_column(event_geo_location, dimension=get_embedding_size(event_geo_location.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_ad_category_id, dimension=get_embedding_size(doc_ad_category_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_ad_topic_id, dimension=get_embedding_size(doc_ad_topic_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_ad_entity_id, dimension=get_embedding_size(doc_ad_entity_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_event_category_id, dimension=get_embedding_size(doc_event_category_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_event_topic_id, dimension=get_embedding_size(doc_event_topic_id.length), combiner='mean'),
  tf.contrib.layers.embedding_column(doc_event_entity_id, dimension=get_embedding_size(doc_event_entity_id.length), combiner='mean'),                        
  '''

  return wide_columns, deep_columns

def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_transformed_reader_input_fn(transformed_metadata,
                                    transformed_data_paths,
                                    batch_size,
                                    mode):
  """Wrap the get input features function to provide the runtime arguments."""
  return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=(
          transformed_data_paths[0] if len(transformed_data_paths) == 1
          else transformed_data_paths),
      training_batch_size=batch_size,
      label_keys=[LABEL_COLUMN],
      reader=gzip_reader_fn,
      key_feature_name=KEY_FEATURE_COLUMN,
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
      num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def get_vocab_sizes():
  """Read vocabulary sizes from the metadata."""
  # TODO(b/35300113) This method will change as we move to tf-transform and use
  # the new schema and statistics protos. For now return a large-ish constant
  # (exact vocabulary size not needed, since we are doing "mod" in tf.Learn).
  # Note that the current workaround might come with a quality sacrifice that
  # should hopefully be lifted soon.
  return {'event_weekend': 2,
          'user_has_already_viewed_doc': 2,
          'event_hour': 6,
          'event_platform': 3,
          'traffic_source': 3,
          'ad_id': 418295,
          'doc_id': 143856,
          'doc_event_id': 636482,
          'ad_advertiser': 2052,
          'doc_ad_source_id': 6339,
          'doc_event_publisher_id': 830,
          'doc_event_source_id': 6339,
          'event_country': 222,
          'event_country_state': 1892,
          'event_geo_location': 2273,
          'doc_ad_category_id': 90,
          'doc_ad_topic_id': 301,
          'doc_ad_entity_id': 52439,
          'doc_event_category_id': 90,
          'doc_event_topic_id': 301,
          'doc_event_entity_id': 52439}


def map_custom_metric(predictions, labels, weights=None, 
                  metrics_collections=None, updates_collections=None,
                  name=None):
  
  display_ids_tf = weights
  #predictions = tf.expand_dims(predictions, -1)
  #labels = tf.expand_dims(labels, -1)

  #Processing unique display_ids, indexes and counts
  display_ids_unique, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(tf.reshape(display_ids_tf, [-1]))
  #print('UNIQUE:', display_ids_unique.get_shape())
  #print('IDX:', display_ids_idx.get_shape())
  #print('COUNT:', display_ids_ads_count.get_shape())

  #Grouping by predictions and labels

  #int(display_ids_tf.get_shape()[0])
  #display_ids_unique_count = tf.shape(display_ids_unique)
  NUM_PARTITIONS=EVAL_BATCH_SIZE//4   
  partitions_preds  = tf.dynamic_partition(predictions, display_ids_idx, NUM_PARTITIONS, name=None)
  partitions_labels = tf.dynamic_partition(labels, display_ids_idx, NUM_PARTITIONS, name=None)

  #max_ads_per_display_id=tf.reduce_max(display_ids_ads_count)
  max_ads_per_display_id=18
  preds_rows = []
  labels_rows = []
  for preds_t, labels_t in zip(partitions_preds, partitions_labels):      
      preds_zero_padding = tf.zeros([max_ads_per_display_id,1], dtype=preds_t.dtype)
      preds_padded =  tf.reshape(tf.concat([preds_t, preds_zero_padding], axis=0)[:max_ads_per_display_id], [1,-1])      
      preds_rows.append(preds_padded)
 
      labels_zero_padding = tf.zeros([max_ads_per_display_id,1], dtype=labels_t.dtype)   
      labels_padded =  tf.reshape(tf.concat([labels_t, labels_zero_padding], axis=0)[:max_ads_per_display_id], [1,-1]) 
      label = tf.argmax(labels_padded, axis=1)
      labels_rows.append(label)  

  preds_matrix = tf.concat(preds_rows, axis=0)[:tf.shape(display_ids_unique)[0],:]
  labels_matrix = tf.concat(labels_rows, axis=0)[:tf.shape(display_ids_unique)[0]]  
  #logging.info("PREDS: %s",preds_matrix.get_shape())
  #logging.info("LABELS:  %s",labels_matrix.get_shape())

  map_value, map_update_op = tf.contrib.metrics.streaming_sparse_average_precision_at_k(
                                predictions=preds_matrix, 
                                labels=labels_matrix, 
                                weights=None, 
                                k=12, 
                                metrics_collections=metrics_collections, 
                                updates_collections=updates_collections, 
                                name="streaming_map")

  return map_value, map_update_op


def map_with_leak_custom_metric(predictions, labels, weights=None, 
                  metrics_collections=None, updates_collections=None,
                  name=None):

  display_ids_tf = tf.to_int64(weights/10)
  is_leak_tf = tf.mod(weights, 10)

  #predictions = tf.expand_dims(predictions, -1)
  #labels = tf.expand_dims(labels, -1)

  #Processing unique display_ids, indexes and counts
  display_ids_unique, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(tf.reshape(display_ids_tf, [-1]))
  #print('UNIQUE:', display_ids_unique.get_shape())
  #print('IDX:', display_ids_idx.get_shape())
  #print('COUNT:', display_ids_ads_count.get_shape())

  #Grouping by predictions and labels

  #int(display_ids_tf.get_shape()[0])
  #display_ids_unique_count = tf.shape(display_ids_unique)
  NUM_PARTITIONS=EVAL_BATCH_SIZE//4   
  partitions_preds  = tf.dynamic_partition(predictions, display_ids_idx, NUM_PARTITIONS, name=None)
  partitions_labels = tf.dynamic_partition(labels, display_ids_idx, NUM_PARTITIONS, name=None)
  partitions_is_leaks  = tf.dynamic_partition(is_leak_tf, display_ids_idx, NUM_PARTITIONS, name=None)

  #max_ads_per_display_id=tf.reduce_max(display_ids_ads_count)
  max_ads_per_display_id=18
  preds_rows = []
  labels_rows = []
  leaks_rows = []
  for preds_t, labels_t, is_leaks_t in zip(partitions_preds, partitions_labels, partitions_is_leaks):  
      #Summing preds with is leak so that the leak (1) has the greatest value
      preds_plus_leak_t = preds_t + tf.to_float(is_leaks_t)

      preds_zero_padding = tf.zeros([max_ads_per_display_id,1], dtype=preds_plus_leak_t.dtype)
      preds_padded =  tf.reshape(tf.concat([preds_plus_leak_t, preds_zero_padding], axis=0)[:max_ads_per_display_id], [1,-1])      
      preds_rows.append(preds_padded)

      labels_zero_padding = tf.zeros([max_ads_per_display_id,1], dtype=labels_t.dtype)   
      labels_padded =  tf.reshape(tf.concat([labels_t, labels_zero_padding], axis=0)[:max_ads_per_display_id], [1,-1]) 
      label = tf.argmax(labels_padded, axis=1)
      labels_rows.append(label)  

  preds_matrix = tf.concat(preds_rows, axis=0)[:tf.shape(display_ids_unique)[0],:]
  labels_matrix = tf.concat(labels_rows, axis=0)[:tf.shape(display_ids_unique)[0]]  
  #logging.info("PREDS: %s",preds_matrix.get_shape())
  #logging.info("LABELS:  %s",labels_matrix.get_shape())

  map_value, map_update_op = tf.contrib.metrics.streaming_sparse_average_precision_at_k(
                                predictions=preds_matrix, 
                                labels=labels_matrix, 
                                weights=None, 
                                k=12, 
                                metrics_collections=metrics_collections, 
                                updates_collections=updates_collections, 
                                name="streaming_map")

  return map_value, map_update_op

def get_experiment_fn(args):
  """Wrap the get experiment function to provide the runtime arguments."""
  vocab_sizes = get_vocab_sizes()
  use_crosses = not args.ignore_crosses

  def get_experiment(output_dir):
    """Function that creates an experiment http://goo.gl/HcKHlT.

    Args:
      output_dir: The directory where the training output should be written.
    Returns:
      A `tf.contrib.learn.Experiment`.
    """

    config = PIPELINE_CONFIG
    wide_columns, deep_columns = get_feature_columns(config, args.model_type, vocab_sizes, use_crosses)

    
    runconfig = tf.contrib.learn.RunConfig()
    cluster = runconfig.cluster_spec
    num_table_shards = max(1, runconfig.num_ps_replicas * 3)
    num_partitions = max(1, 1 + cluster.num_tasks('worker') if cluster and
                         'worker' in cluster.jobs else 0)
    

    l1_regularization = args.l1_regularization or config[L1_REGULARIZATION]
    l2_regularization = args.l2_regularization or config[L2_REGULARIZATION]

    if args.model_type == WIDE:
      '''
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 0.1
      learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                 10000, 0.96, staircase=True)
      '''

      estimator = tf.contrib.learn.LinearClassifier(
          model_dir=output_dir,
          feature_columns=wide_columns,
          optimizer=tf.train.FtrlOptimizer(
                                      learning_rate=0.05,
                                      l1_regularization_strength=0.1,
                                      l2_regularization_strength=0.0)
                                      #l1_regularization_strength=l1_regularization,
                                      #l2_regularization_strength=l2_regularization)
                                      )
          #optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
          #    example_id_column=KEY_FEATURE_COLUMN,
          #    symmetric_l1_regularization=0.1,
          #    #symmetric_l2_regularization=1.0,
          #    num_loss_partitions=num_partitions,  # workers
          #    num_table_shards=num_table_shards))  # ps
    elif args.model_type == DEEP:
      estimator = tf.contrib.learn.DNNClassifier(
          hidden_units=args.hidden_units,
          feature_columns=deep_columns,
          model_dir=output_dir,
          dropout=0.1,
          #optimizer=tf.train.AdagradOptimizer(
          #  #learning_rate=0.05, 
          #  learning_rate=0.1, 
          #  initial_accumulator_value=0.1, 
          #  use_locking=False)
          #https://www.tensorflow.org/api_docs/python/tf/train/ProximalAdagradOptimizer
          optimizer=tf.train.ProximalAdagradOptimizer(
              learning_rate=0.05,
              #learning_rate=0.1,
              initial_accumulator_value=0.1,              
              l1_regularization_strength=0.0, 
              l2_regularization_strength=1e-3,
              use_locking=False
            )
          )
     

    elif args.model_type == WIDE_N_DEEP:
      estimator = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=output_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=tf.train.FtrlOptimizer(
                                                    learning_rate=0.05,
                                                    l1_regularization_strength=0.1,
                                                    l2_regularization_strength=0.0),                                          
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=args.hidden_units,
            dnn_dropout=0.1,
            dnn_optimizer=tf.train.ProximalAdagradOptimizer(
              learning_rate=0.05,
              initial_accumulator_value=0.1,              
              l1_regularization_strength=0.0, 
              l2_regularization_strength=1e-3,
              use_locking=False
            ))

    transformed_metadata = metadata_io.read_metadata(
        args.transformed_metadata_path)
    raw_metadata = metadata_io.read_metadata(args.raw_metadata_path)
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata,
            args.transform_savedmodel,
            raw_label_keys=[LABEL_COLUMN]))
    export_strategy = (
        tf.contrib.learn.utils.make_export_strategy(
            serving_input_fn, exports_to_keep=5,
            default_output_alternative_key=None))

    train_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, args.train_data_paths, args.batch_size,
        tf.contrib.learn.ModeKeys.TRAIN)

    eval_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, args.eval_data_paths, args.eval_batch_size,
        tf.contrib.learn.ModeKeys.EVAL)

    train_set_size = args.train_set_size or config[NUM_EXAMPLES]

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_steps=(args.train_steps or
                     args.num_epochs * train_set_size // args.batch_size),
        #eval_steps=args.eval_steps,
        eval_steps = (args.eval_steps or
                      int(math.ceil(args.eval_set_size / float(args.eval_batch_size)))),        
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,    
        min_eval_frequency=2000,    
        #min_eval_frequency=None,   
        export_strategies=export_strategy,     
        eval_metrics={
        'MAP':
            metric_spec.MetricSpec(
                metric_fn=map_custom_metric,
                prediction_key="logistic", #probabilities, logits, logistic, classes
                #label_key="label",
                weight_key=DISPLAY_ID_COLUMN
                ), 
        'MAP with Leak':
            metric_spec.MetricSpec(
                metric_fn=map_with_leak_custom_metric,
                prediction_key="logistic", #probabilities, logits, logistic, classes
                #label_key="label",
                weight_key=DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN
                )
        }
        )

  # Return a function to create an Experiment.
  return get_experiment


def main(argv=None):
  """Run a Tensorflow model on the Criteo dataset."""
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  learn_runner.run(experiment_fn=get_experiment_fn(args),
                   output_dir=output_dir)


if __name__ == '__main__':
  main()