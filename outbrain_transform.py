# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Schema and tranform definition for the Criteo dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import coders
from tensorflow_transform.tf_metadata import dataset_schema

from trainer.features import LABEL_COLUMN, DISPLAY_ID_COLUMN, AD_ID_COLUMN, IS_LEAK_COLUMN, DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN, CATEGORICAL_COLUMNS, DOC_CATEGORICAL_MULTIVALUED_COLUMNS, BOOL_COLUMNS, INT_COLUMNS, FLOAT_COLUMNS, FLOAT_COLUMNS_LOG_BIN_TRANSFORM, FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM



#SINGLE_VALUED_COLUMNS = CATEGORICAL_COLUMNS + BOOL_COLUMNS + INT_COLUMNS + FLOAT_COLUMNS


CSV_ORDERED_COLUMNS = ['label','display_id','ad_id','doc_id','doc_event_id','is_leak','event_weekend',
              'user_has_already_viewed_doc','user_views','ad_views','doc_views',
              'doc_event_days_since_published','doc_event_hour','doc_ad_days_since_published',
              'doc_views_cf','doc_distinct_users_cf','user_views_cf','user_distinct_docs_cf',
              'ubcf_als_score_cf','ibcf_als_score_cf','doc_avg_views_by_distinct_users_cf',
              'user_avg_views_of_distinct_docs_cf','pop_ad_id','pop_ad_id_conf',
              'pop_ad_id_conf_multipl','pop_document_id','pop_document_id_conf',
              'pop_document_id_conf_multipl','pop_publisher_id','pop_publisher_id_conf',
              'pop_publisher_id_conf_multipl','pop_advertiser_id','pop_advertiser_id_conf',
              'pop_advertiser_id_conf_multipl','pop_campain_id','pop_campain_id_conf',
              'pop_campain_id_conf_multipl','pop_doc_event_doc_ad','pop_doc_event_doc_ad_conf',
              'pop_doc_event_doc_ad_conf_multipl','pop_source_id','pop_source_id_conf',
              'pop_source_id_conf_multipl','pop_source_id_country','pop_source_id_country_conf',
              'pop_source_id_country_conf_multipl','pop_entity_id','pop_entity_id_conf',
              'pop_entity_id_conf_multipl','pop_entity_id_country','pop_entity_id_country_conf',
              'pop_entity_id_country_conf_multipl','pop_topic_id','pop_topic_id_conf',
              'pop_topic_id_conf_multipl','pop_topic_id_country','pop_topic_id_country_conf',
              'pop_topic_id_country_conf_multipl','pop_category_id','pop_category_id_conf',
              'pop_category_id_conf_multipl','pop_category_id_country','pop_category_id_country_conf',
              'pop_category_id_country_conf_multipl','user_doc_ad_sim_categories',
              'user_doc_ad_sim_categories_conf','user_doc_ad_sim_categories_conf_multipl',
              'user_doc_ad_sim_topics','user_doc_ad_sim_topics_conf','user_doc_ad_sim_topics_conf_multipl',
              'user_doc_ad_sim_entities','user_doc_ad_sim_entities_conf','user_doc_ad_sim_entities_conf_multipl',
              'doc_event_doc_ad_sim_categories','doc_event_doc_ad_sim_categories_conf',
              'doc_event_doc_ad_sim_categories_conf_multipl','doc_event_doc_ad_sim_topics',
              'doc_event_doc_ad_sim_topics_conf','doc_event_doc_ad_sim_topics_conf_multipl',
              'doc_event_doc_ad_sim_entities','doc_event_doc_ad_sim_entities_conf',
              'doc_event_doc_ad_sim_entities_conf_multipl','ad_advertiser','doc_ad_category_id_1',
              'doc_ad_category_id_2','doc_ad_category_id_3','doc_ad_topic_id_1','doc_ad_topic_id_2',
              'doc_ad_topic_id_3','doc_ad_entity_id_1','doc_ad_entity_id_2','doc_ad_entity_id_3',
              'doc_ad_entity_id_4','doc_ad_entity_id_5','doc_ad_entity_id_6','doc_ad_publisher_id',
              'doc_ad_source_id','doc_event_category_id_1','doc_event_category_id_2','doc_event_category_id_3',
              'doc_event_topic_id_1','doc_event_topic_id_2','doc_event_topic_id_3','doc_event_entity_id_1',
              'doc_event_entity_id_2','doc_event_entity_id_3','doc_event_entity_id_4','doc_event_entity_id_5',
              'doc_event_entity_id_6','doc_event_publisher_id','doc_event_source_id','event_country',
              'event_country_state','event_geo_location','event_hour','event_platform','traffic_source']


def make_csv_coder(schema, mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Produces a CsvCoder (with tab as the delimiter) from a data schema.

  Args:
    schema: A tf.Transform `Schema` object.
    mode: tf.contrib.learn.ModeKeys specifying if the source is being used for
      train/eval or prediction.

  Returns:
    A tf.Transform CsvCoder.
  """

  column_names = CSV_ORDERED_COLUMNS
  if mode == tf.contrib.learn.ModeKeys.INFER:
    column_names.remove(LABEL_COLUMN)

  return coders.CsvCoder(column_names, schema, delimiter=',')


def make_input_schema(mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Input schema definition.

  Args:
    mode: tf.contrib.learn.ModeKeys specifying if the schema is being used for
      train/eval or prediction.
  Returns:
    A `Schema` object.
  """
  #result = ({} if mode == tf.contrib.learn.ModeKeys.INFER
  #          else {LABEL_COLUMN: tf.FixedLenFeature(shape=[], dtype=tf.int64)})

  result = {}
  result[LABEL_COLUMN] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
  result[DISPLAY_ID_COLUMN] = tf.FixedLenFeature(shape=[], dtype=tf.float32)
  #result[AD_ID_COLUMN] = tf.VarLenFeature(dtype=tf.float32)
  result[IS_LEAK_COLUMN] = tf.FixedLenFeature(shape=[], dtype=tf.int64)

  for name in BOOL_COLUMNS:
    result[name] = tf.VarLenFeature(dtype=tf.int64)
  #TODO: Create dummy features that indicates whether any of the numeric features is null 
  #(currently default 0 value might introduce noise)
  for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM+FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
    result[name] = tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0)  
  for name in INT_COLUMNS:
    result[name] = tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0)
  for name in CATEGORICAL_COLUMNS:
    result[name] = tf.VarLenFeature(dtype=tf.float32)
  for multi_category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
    for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category]:
      result[category] = tf.VarLenFeature(dtype=tf.float32)

  return dataset_schema.from_feature_spec(result)

def tf_log2_1p(x):
  return tf.log1p(x) / tf.log(2.0)

#def make_preprocessing_fn(frequency_threshold):
def make_preprocessing_fn():
  """Creates a preprocessing function for criteo.

  Args:
    frequency_threshold: The frequency_threshold used when generating
      vocabularies for the categorical features.

  Returns:
    A preprocessing function.
  """
  def preprocessing_fn(inputs):
    """User defined preprocessing function for criteo columns.

    Args:
      inputs: dictionary of input `tensorflow_transform.Column`.
    Returns:
      A dictionary of `tensorflow_transform.Column` representing the transformed
          columns.
    """
    # TODO(b/35001605) Make this "passthrough" more DRY.
    result = {LABEL_COLUMN: tft.map(lambda x: tf.expand_dims(x, -1), inputs[LABEL_COLUMN]), 
              DISPLAY_ID_COLUMN: tft.map(lambda x: tf.expand_dims(tf.to_int64(x), -1), inputs[DISPLAY_ID_COLUMN]),
              IS_LEAK_COLUMN: tft.map(lambda x: tf.expand_dims(x, -1), inputs[IS_LEAK_COLUMN]),
              DISPLAY_ID_AND_IS_LEAK_ENCODED_COLUMN: tft.map(lambda display_id, is_leak: tf.expand_dims((tf.to_int64(display_id)*10)+tf.nn.relu(is_leak), -1), inputs[DISPLAY_ID_COLUMN], inputs[IS_LEAK_COLUMN])}


    for name in FLOAT_COLUMNS:
      result[name] = tft.map(lambda x: tf.expand_dims(x, -1), inputs[name])

    #For well-distributed percentages, creating 10 bins
    for name in FLOAT_COLUMNS_SIMPLE_BIN_TRANSFORM:
      result[name+'_binned'] = tft.map(lambda x: tf.expand_dims(tf.to_int64(x*10), -1), inputs[name])

    #For log-distributed percentages, creating bins on log
    for name in FLOAT_COLUMNS_LOG_BIN_TRANSFORM:
      result[name+'_log_binned'] = tft.map(lambda x: tf.expand_dims(tf.to_int64(tf_log2_1p(x*1000)), -1), inputs[name])
      result[name+'_log_01scaled'] = tft.scale_to_0_1(tft.map(lambda x: tf.expand_dims(tf_log2_1p(x*1000), -1), inputs[name]))

    #Apply the log to smooth high counts (outliers) and scale from 0 to 1
    for name in INT_COLUMNS:    
      result[name+'_log_int']  = tft.map(lambda x: tf.expand_dims(tf.to_int64(tf_log2_1p(x)), -1), inputs[name])
      result[name+'_log_01scaled'] = tft.scale_to_0_1(tft.map(lambda x: tf.expand_dims(tf_log2_1p(x), -1), inputs[name]))
      #result[name] = tft.map(lambda x: tf.expand_dims(tf.to_int64(x), -1), inputs[name])
    
    #for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS + \
    #            [category for multicategory in DOC_CATEGORICAL_MULTIVALUED_COLUMNS for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multicategory]]:
    for name in BOOL_COLUMNS + CATEGORICAL_COLUMNS:
      result[name] = tft.map(lambda x: tf.to_int64(x), inputs[name])

    #result['display_ad_id_key'] = tft.map(lambda display_id, ad_id: tf.multiply(tf.sparse_tensor_to_dense(tf.to_int64(display_id)), int(1e8)) + tf.sparse_tensor_to_dense(tf.to_int64(ad_id)), inputs['display_id'], inputs['ad_id'])

    #result['doc_ad_category_id'] = tft.map(lambda cat1, cat2, cat3: tf.to_int64(tf.sparse_concat(axis=1, sp_inputs=[cat1, cat2, cat3])), 
    #                                       inputs['doc_ad_category_id_1'], inputs['doc_ad_category_id_2'], inputs['doc_ad_category_id_3'])
    

    for multicategory in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
      if len(DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multicategory]) == 3:
        result[multicategory] = tft.map(lambda col1, col2, col3: tf.to_int64(tf.sparse_concat(axis=1, sp_inputs=[col1, col2, col3])), 
                                        *[inputs[category] for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multicategory]])
      elif len(DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multicategory]) == 6:
        result[multicategory] = tft.map(lambda col1, col2, col3, col4, col5, col6: tf.to_int64(tf.sparse_concat(axis=1, sp_inputs=[col1, col2, col3, col4, col5, col6])), 
                                        *[inputs[category] for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multicategory]])


    #TODO: 
    #Ref: https://github.com/tensorflow/transform/blob/master/getting_started.md
    #tf.stack([cat1, cat2, cat3])
    #[inputs[name] for category in zip(*DOC_CATEGORICAL_MULTIVALUED_COLUMNS['doc_ad_category_id']) ]

    
    #for multi_category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS:
    #  result[multi_category] = tft.map(lambda x: tf.stack(x), [inputs[category] for category in DOC_CATEGORICAL_MULTIVALUED_COLUMNS[multi_category] ])      
    

    #for name in CATEGORICAL_COLUMN_NAMES:
    #  result[name + '_id'] = tft.string_to_int(
    #      inputs[name], frequency_threshold=frequency_threshold)

    # TODO(b/35318962): Obviate the need for this workaround on Dense features.
    # FeatureColumns expect shape (batch_size, 1), not just (batch_size)
    #result = {
    #    k: tft.map(lambda x: tf.expand_dims(x, -1), v)
    #    for k, v in result.items()
    #}


    return result

  return preprocessing_fn
