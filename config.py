# names of the columns of the input dataset, in this order
COLUMN_NAMES = [
    'event_id',
    'tenant_id',
    'user_agent',
    'operation_type',
    'operation_result',
    'operation_description',
    'source_ip',
    'app_id',
    'uid',
    'day',
    'hour',
    'minute',
    'second',
    'risk_label',
    'weight_column'
]

# datatypes of the column in the order of COLUMN_NAMES
COLUMN_DTYPES = [
    'str',
    'str',
    'str',
    'str',
    'str',
    'str',
    'str',
    'str',
    'str',
    'int',
    'int',
    'int',
    'int',
    'int',
    'int',
]

# the categorical features
CATEGORICAL_FEATURES = [
    'event_id',
    'tenant_id',
    'user_agent',
    'operation_type',
    'operation_result',
    'operation_description',
    'source_ip',
    'app_id',
    'uid'
]

# the continuous features
CONTINUOUS_FEATURES = [
    'day',
    'hour',
    'minute',
    'second'
]

CROSS_COLUMNS = None    # if None 3 random cross columns will be chosen

WEIGHT_COLUMN = 'weight_column'

LABEL = 'risk_label' 

HIDDEN_LAYERS = [256, 128, 64]

EMBEDDING_DIMENSIONS = {
    'event_id': 10,
    'tenant_id': 10,
    'user_agent': 10,
    'operation_type': 10,
    'operation_result': 10,
    'operation_description': 10,
    'source_ip': 10,
    'app_id': 10,
    'uid': 10
}

DROPOUT = 0

TRAIN_PATH = '' # What is the path

TEST_PATH = '' # what is the test path

WITH_WEIGHT = False

EPOCHS = 1

VERBOSE = True
