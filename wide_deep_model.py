import tensorflow as tf
import time
import pprint
import random
import turicreate as tc
from functools import reduce
from config import COLUMN_NAMES, COLUMN_DTYPES, CATEGORICAL_FEATURES
from config import CONTINUOUS_FEATURES, WEIGHT_COLUMN, LABEL
from config import HIDDEN_LAYERS, EMBEDDING_DIMENSIONS, DROPOUT
from config import TRAIN_PATH, TEST_PATH, WITH_WEIGHT, EPOCHS, VERBOSE
from config import CROSS_COLUMN, MODEL_DIR

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(10)
pp = pprint.PrettyPrinter()

def compute_all_crosses(categoricals):
    def get_2_combos(features):
        features = features.copy()
        result = []
        for i in range(len(features)):
            feat = features.pop(0)
            for f in features:
                result.append([feat,f])
        return result
    
    def get_more_combos(features, two_features):
        features = features.copy()
        result = []
        feature_set = set(features)

        for combo in two_features:
            combo_set = set(combo)
            remain_set = feature_set - combo_set

            for feat in remain_set:
                temp_combo = combo_set.copy()
                temp_combo.add(feat)
                result.append(sorted(list(temp_combo)))
            
        return sorted(result, key=lambda x:x[0][0]+x[1][0]+x[2][0])

    def remove_dups(crosses):
        count = {}
        crosses_copy = crosses.copy()
        for cross in crosses:
            h = [hash(x) for x in cross]
            h = reduce(lambda x,y: x+y, h)
            if h not in count:
                count[h] = True
            else:
                crosses_copy.remove(cross)
        return crosses_copy

    two_feats = get_2_combos(categoricals)
    results = {'2 crosses': two_feats}
    
    for i in range(len(categoricals)-2):
        next_combo = get_more_combos(categoricals, results[f'{i+2} crosses'])
        next_combo = remove_dups(next_combo)
        results[f'{i+3} crosses'] = next_combo
    
    return results

ALL_COLUMN_CROSSES_DICT = compute_all_crosses(CATEGORICAL_FEATURES)

ALL_COLUMN_CROSSES = reduce(lambda x,y: x+y, [item for _, item in ALL_COLUMN_CROSSES_DICT.items()])

def get_dataset(file_path, with_weight=True, epochs=10, batch_size=128):
    """Gets a tf.data.dataset pipeline for training and evaluation
    """

    FIELD_DEFAULTS = [[''] if t=='str' else [0.] for t in COLUMN_DTYPES]
    data_set = tf.data.TextLineDataset(file_path).skip(1)

    def _parse_line(line):
        fields = tf.decode_csv(line, FIELD_DEFAULTS)
        features = dict(zip(COLUMN_NAMES, fields))
        label = features.pop(LABEL)
        if with_weight:
            features.pop('weight_column')
        return features, label

    data_set = data_set.map(_parse_line)

    data_set = data_set.repeat(epochs).shuffle(1000).batch(batch_size)

    return data_set.make_one_shot_iterator().get_next()

def create_model(cross_columns, hidden_units, embedding_dim, 
                 dropout=0, with_weight=True, verbose=True):
    """Creates an instance of DNNLinearCombinedClassifier with the given cross columns
    
    Args:
        cross_columns (list): elements from ALL_CROSS_COLUMNS
        hidden_units (list): the number of hidden units for each layer
        embedding_dim (dict): the dimensions for each categorical feature
        dropout (float): a number between 0 and 1 for the dropout rate
        with_weight (bool): to control wether to train with weighted loss
        verbose (bool): control wether to print verbosely
    
    returns:
        a wide and deep model instance
        
    usage:
        model = create_model(
                    cross_column=[['event_id', 'tenant_id'], ['operation_type', 'app_id']],
                    hidden_units=[256, 132, 64],
                    embedding_dim={'event_id': 5
                                   'tenant_id': 5
                                   'user_agent': 5
                                   'operation_type': 5
                                   'operation_result': 5
                                   'operation_description': 5
                                   'source_ip': 5
                                   'app_id': 5
                                   'uid': 5},
                     dropout=0,
                     with_weight=False,
                     verbose=True)
    """
    
    if verbose:
        text = {}
        text['embedding dimensions'] = embedding_dim
        text['cross columns'] = cross_columns
        text['hidden units'] = hidden_units
        text['dropout'] = dropout
        text['with weight'] = with_weight
        pp.pprint({'creating model': text})

    # Continuous columns dict['name'] = tf.feature_column
    continuous_columns = {
        name: tf.feature_column.numeric_column(name)
        for name in CONTINUOUS_FEATURES
    }
    
    # Categorical columns dict['name'] = tf.feature_column
    categorical_columns = {
        name: tf.feature_column.categorical_column_with_hash_bucket(
                name, hash_bucket_size = 10000) # this is a hyperparameter
        for name in CATEGORICAL_FEATURES
    }
    
    base_columns = [col for _, col in categorical_columns.items()]
    
    crossed_columns = [
        tf.feature_column.crossed_column(cross, hash_bucket_size=10000)
        for cross in cross_columns
    ]
    
    deep_columns = [col for _, col in continuous_columns.items()]+[
        tf.feature_column.embedding_column(col, dimension=embedding_dim[name])
        for name, col in categorical_columns.items()
    ]
    
    if with_weight:
        weight_column = tf.feature_column.numeric_column(WEIGHT_COLUMN)
    else:
        weight_column = None
        

    model = tf.estimator.DNNLinearCombinedClassifier(
                model_dir = MODEL_DIR,
                linear_feature_columns=base_columns + crossed_columns,
                dnn_feature_columns=deep_columns,
                weight_column=weight_column,
                dnn_hidden_units=hidden_units
            )

    return model

def train_model(model, train_path, with_weight=True, epochs=1, verbose=True):
    """Trains the model with given data path
    
    Args:
        model: a tf.estimator instance to be trained
        train_path (str): the path to the training data
        with_weight (bool): to control training with weighted loss
        epochs (int): number of epochs to train
        verbose (bool): control verbose printing
    
    returns:
        None
    """
    # TRAINING BLOCK #
    if verbose:
        print('training model')
    train_start = time.time()
    model.train(input_fn=lambda:get_dataset(train_path, with_weight=with_weight, epochs=epochs)) 
    train_end = time.time()
    print(f'training time: {train_end-train_start}')
    # TRAINING BLOCK #
    
    
def evaluate_model(model, test_path, with_weight=True, verbose=True):
    """Evaluates the given model
    
    Args:
        model: a tf.estimator instance to be evaluated
        test_path (str): the path to the testing data
        with_weight (bool): wether the model was trained with weighted loss
        verbose (bool): controls verbose printing
    returns:
        dict of evaluation metrics and their scores
    """
    # EVALUTATION BLOCK #
    if verbose:
        print('evaluating model')
    eval_start = time.time()
    evaluation = model.evaluate(input_fn=lambda:get_dataset(test_path, with_weight=with_weight, epochs=1))
    eval_end = time.time()
    print(f'evaluate time: {eval_end-eval_start}')
    # EVALUATION BLOCK #
    return evaluation

def run_model(train_path, test_path, column_crosses, hidden_units, 
              embedding_dim, with_weight=True, epochs=1, verbose=True):
    """Trains and evaluates an instance of a model with tf.data.datasets
    
    Args: 
        train_path (str): file path to training data
        test_path (str): file path to testing data
        cross_columns (list): elements from ALL_CROSS_COLUMNS
        hidden_units (list): the number of hidden units for each layer
        embedding_dim (dict): the dimensions for each categorical feature
        dropout (float): a number between 0 and 1 for the dropout rate
        with_weight (bool): to control wether to train with weighted loss
        epochs (int): number of epochs to train
        verbose (bool): control wether to print verbosely
    
    returns:
        dict of evaluation metrics and their scores for the model
    """
    model = create_model(column_crosses, hidden_units, with_weight=with_weight, 
                         verbose=verbose, embedding_dim=embedding_dim)

    train_model(model, train_path, with_weight=with_weight, epochs=epochs, verbose=verbose)

    evaluation = evaluate_model(model, test_path, with_weight=with_weight, verbose=verbose)
    
    return evaluation

if __name__ == '__main__':
    if CROSS_COLUMN is None:
        CROSS_COLUMN = random.sample(ALL_COLUMN_CROSSES, 3)
    run_model(TRAIN_PATH, TEST_PATH, CROSS_COLUMN, 
            HIDDEN_LAYERS, EMBEDDING_DIMENSIONS, 
            with_weight=WITH_WEIGHT,epochs=EPOCHS, verbose=VERBOSE)
