##### FEATURE SETTINGS #####
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

# if None 3 random cross columns will be chosen
CROSS_COLUMNS = None    

# if CROSS_COLUMN choose this many random cross columns
NUM_RANDOM_CROSSES = 3  

# name of weight column for weighted loss
WEIGHT_COLUMN = 'weight_column'

# name of label column
LABEL = 'risk_label' 


##### MODEL PARAMETERS ######
HIDDEN_LAYERS = [256, 128, 64]

# embedding dimesions for each feature in the deep network
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

# dropout rate
DROPOUT = 0

# train with weighted loss
WITH_WEIGHT = False

# number of epochs to train
EPOCHS = 1

# Controls verbose printing
VERBOSE = True

##### FILE PATHS #####
TRAIN_PATH = '' # What is the path

TEST_PATH = '' # what is the test path

# Warning: 
# if a model was trained with a random selection of 
# cross columns, training for a second time will give 
# an error if the cross columns do not match the last
# instance of the model
MODEL_DIR = 'model/'




