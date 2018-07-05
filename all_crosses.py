ALL_COLUMN_CROSSES_DICT = {
    '2 crosses':[
    ['event_id', 'tenant_id'],
    ['event_id', 'user_agent'],
    ['event_id', 'operation_type'],
    ['event_id', 'operation_result'],
    ['event_id', 'operation_description'],
    ['event_id', 'source_ip'],
    ['event_id', 'app_id'],
    ['event_id', 'uid'],
    ['tenant_id', 'user_agent'],
    ['tenant_id', 'operation_type'],
    ['tenant_id', 'operation_result'],
    ['tenant_id', 'operation_description'],
    ['tenant_id', 'source_ip'],
    ['tenant_id', 'app_id'],
    ['tenant_id', 'uid'],
    ['user_agent', 'operation_type'],
    ['user_agent', 'operation_result'],
    ['user_agent', 'operation_description'],
    ['user_agent', 'source_ip'],
    ['user_agent', 'app_id'],
    ['user_agent', 'uid'],
    ['operation_type', 'operation_result'],
    ['operation_type', 'operation_description'],
    ['operation_type', 'source_ip'],
    ['operation_type', 'app_id'],
    ['operation_type', 'uid'],
    ['operation_result', 'operation_description'],
    ['operation_result', 'source_ip'],
    ['operation_result', 'app_id'],
    ['operation_result', 'uid'],
    ['operation_description', 'source_ip'],
    ['operation_description', 'app_id'],
    ['operation_description', 'uid'],
    ['source_ip', 'app_id'],
    ['source_ip', 'uid'],
    ['app_id', 'uid']],

    '3 crosses':[
    ['app_id', 'event_id', 'operation_type'],
    ['app_id', 'event_id', 'operation_result'],
    ['app_id', 'event_id', 'operation_description'],
    ['app_id', 'event_id', 'source_ip'],
    ['app_id', 'event_id', 'tenant_id'],
    ['app_id', 'event_id', 'user_agent'],
    ['app_id', 'event_id', 'uid'],
    ['app_id', 'operation_result', 'operation_type'],
    ['app_id', 'operation_description', 'operation_result'],
    ['app_id', 'operation_description', 'operation_type'],
    ['app_id', 'operation_result', 'source_ip'],
    ['app_id', 'operation_description', 'source_ip'],
    ['app_id', 'operation_type', 'source_ip'],
    ['app_id', 'operation_type', 'tenant_id'],
    ['app_id', 'operation_result', 'tenant_id'],
    ['app_id', 'operation_description', 'tenant_id'],
    ['app_id', 'operation_type', 'user_agent'],
    ['app_id', 'operation_result', 'user_agent'],
    ['app_id', 'operation_description', 'user_agent'],
    ['app_id', 'operation_result', 'uid'],
    ['app_id', 'operation_description', 'uid'],
    ['app_id', 'operation_type', 'uid'],
    ['app_id', 'source_ip', 'tenant_id'],
    ['app_id', 'source_ip', 'user_agent'],
    ['app_id', 'source_ip', 'uid'],
    ['app_id', 'tenant_id', 'user_agent'],
    ['app_id', 'tenant_id', 'uid'],
    ['app_id', 'uid', 'user_agent'],
    ['event_id', 'operation_result', 'operation_type'],
    ['event_id', 'operation_description', 'operation_type'],
    ['event_id', 'operation_description', 'operation_result'],
    ['event_id', 'operation_type', 'source_ip'],
    ['event_id', 'operation_result', 'source_ip'],
    ['event_id', 'operation_description', 'source_ip'],
    ['event_id', 'operation_type', 'tenant_id'],
    ['event_id', 'operation_result', 'tenant_id'],
    ['event_id', 'operation_description', 'tenant_id'],
    ['event_id', 'operation_type', 'user_agent'],
    ['event_id', 'operation_result', 'user_agent'],
    ['event_id', 'operation_description', 'user_agent'],
    ['event_id', 'operation_type', 'uid'],
    ['event_id', 'operation_result', 'uid'],
    ['event_id', 'operation_description', 'uid'],
    ['event_id', 'source_ip', 'tenant_id'],
    ['event_id', 'source_ip', 'user_agent'],
    ['event_id', 'source_ip', 'uid'],
    ['event_id', 'tenant_id', 'user_agent'],
    ['event_id', 'tenant_id', 'uid'],
    ['event_id', 'uid', 'user_agent'],
    ['operation_description', 'operation_result', 'operation_type'],
    ['operation_result', 'operation_type', 'source_ip'],
    ['operation_description', 'operation_result', 'source_ip'],
    ['operation_description', 'operation_type', 'source_ip'],
    ['operation_result', 'operation_type', 'tenant_id'],
    ['operation_description', 'operation_type', 'tenant_id'],
    ['operation_description', 'operation_result', 'tenant_id'],
    ['operation_result', 'operation_type', 'user_agent'],
    ['operation_description', 'operation_type', 'user_agent'],
    ['operation_description', 'operation_result', 'user_agent'],
    ['operation_result', 'operation_type', 'uid'],
    ['operation_description', 'operation_result', 'uid'],
    ['operation_description', 'operation_type', 'uid'],
    ['operation_type', 'source_ip', 'tenant_id'],
    ['operation_result', 'source_ip', 'tenant_id'],
    ['operation_description', 'source_ip', 'tenant_id'],
    ['operation_type', 'source_ip', 'user_agent'],
    ['operation_result', 'source_ip', 'user_agent'],
    ['operation_description', 'source_ip', 'user_agent'],
    ['operation_result', 'source_ip', 'uid'],
    ['operation_description', 'source_ip', 'uid'],
    ['operation_type', 'source_ip', 'uid'],
    ['operation_type', 'tenant_id', 'user_agent'],
    ['operation_result', 'tenant_id', 'user_agent'],
    ['operation_description', 'tenant_id', 'user_agent'],
    ['operation_type', 'tenant_id', 'uid'],
    ['operation_result', 'tenant_id', 'uid'],
    ['operation_description', 'tenant_id', 'uid'],
    ['operation_type', 'uid', 'user_agent'],
    ['operation_result', 'uid', 'user_agent'],
    ['operation_description', 'uid', 'user_agent'],
    ['source_ip', 'tenant_id', 'user_agent'],
    ['source_ip', 'tenant_id', 'uid'],
    ['source_ip', 'uid', 'user_agent'],
    ['tenant_id', 'uid', 'user_agent']],
    
    '4 crosses':[
    ['app_id', 'event_id', 'operation_result', 'operation_type'],
    ['app_id', 'event_id', 'operation_description', 'operation_type'],
    ['app_id', 'event_id', 'operation_description', 'operation_result'],
    ['app_id', 'event_id', 'operation_type', 'source_ip'],
    ['app_id', 'event_id', 'operation_result', 'source_ip'],
    ['app_id', 'event_id', 'operation_description', 'source_ip'],
    ['app_id', 'event_id', 'operation_type', 'tenant_id'],
    ['app_id', 'event_id', 'operation_result', 'tenant_id'],
    ['app_id', 'event_id', 'operation_description', 'tenant_id'],
    ['app_id', 'event_id', 'operation_type', 'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'user_agent'],
    ['app_id', 'event_id', 'operation_description', 'user_agent'],
    ['app_id', 'event_id', 'operation_type', 'uid'],
    ['app_id', 'event_id', 'operation_result', 'uid'],
    ['app_id', 'event_id', 'operation_description', 'uid'],
    ['app_id', 'event_id', 'source_ip', 'tenant_id'],
    ['app_id', 'event_id', 'source_ip', 'user_agent'],
    ['app_id', 'event_id', 'source_ip', 'uid'],
    ['app_id', 'event_id', 'tenant_id', 'user_agent'],
    ['app_id', 'event_id', 'tenant_id', 'uid'],
    ['app_id', 'event_id', 'uid', 'user_agent'],
    ['app_id', 'operation_description', 'operation_result', 'operation_type'],
    ['app_id', 'operation_result', 'operation_type', 'source_ip'],
    ['app_id', 'operation_description', 'operation_result', 'source_ip'],
    ['app_id', 'operation_description', 'operation_type', 'source_ip'],
    ['app_id', 'operation_result', 'operation_type', 'tenant_id'],
    ['app_id', 'operation_description', 'operation_type', 'tenant_id'],
    ['app_id', 'operation_description', 'operation_result', 'tenant_id'],
    ['app_id', 'operation_result', 'operation_type', 'user_agent'],
    ['app_id', 'operation_description', 'operation_type', 'user_agent'],
    ['app_id', 'operation_description', 'operation_result', 'user_agent'],
    ['app_id', 'operation_result', 'operation_type', 'uid'],
    ['app_id', 'operation_description', 'operation_result', 'uid'],
    ['app_id', 'operation_description', 'operation_type', 'uid'],
    ['app_id', 'operation_type', 'source_ip', 'tenant_id'],
    ['app_id', 'operation_result', 'source_ip', 'tenant_id'],
    ['app_id', 'operation_description', 'source_ip', 'tenant_id'],
    ['app_id', 'operation_type', 'source_ip', 'user_agent'],
    ['app_id', 'operation_result', 'source_ip', 'user_agent'],
    ['app_id', 'operation_description', 'source_ip', 'user_agent'],
    ['app_id', 'operation_result', 'source_ip', 'uid'],
    ['app_id', 'operation_description', 'source_ip', 'uid'],
    ['app_id', 'operation_type', 'source_ip', 'uid'],
    ['app_id', 'operation_type', 'tenant_id', 'user_agent'],
    ['app_id', 'operation_result', 'tenant_id', 'user_agent'],
    ['app_id', 'operation_description', 'tenant_id', 'user_agent'],
    ['app_id', 'operation_type', 'tenant_id', 'uid'],
    ['app_id', 'operation_result', 'tenant_id', 'uid'],
    ['app_id', 'operation_description', 'tenant_id', 'uid'],
    ['app_id', 'operation_type', 'uid', 'user_agent'],
    ['app_id', 'operation_result', 'uid', 'user_agent'],
    ['app_id', 'operation_description', 'uid', 'user_agent'],
    ['app_id', 'source_ip', 'tenant_id', 'user_agent'],
    ['app_id', 'source_ip', 'tenant_id', 'uid'],
    ['app_id', 'source_ip', 'uid', 'user_agent'],
    ['app_id', 'tenant_id', 'uid', 'user_agent'],
    ['event_id', 'operation_description', 'operation_result', 'operation_type'],
    ['event_id', 'operation_result', 'operation_type', 'source_ip'],
    ['event_id', 'operation_description', 'operation_result', 'source_ip'],
    ['event_id', 'operation_description', 'operation_type', 'source_ip'],
    ['event_id', 'operation_result', 'operation_type', 'tenant_id'],
    ['event_id', 'operation_description', 'operation_type', 'tenant_id'],
    ['event_id', 'operation_description', 'operation_result', 'tenant_id'],
    ['event_id', 'operation_result', 'operation_type', 'user_agent'],
    ['event_id', 'operation_description', 'operation_type', 'user_agent'],
    ['event_id', 'operation_description', 'operation_result', 'user_agent'],
    ['event_id', 'operation_result', 'operation_type', 'uid'],
    ['event_id', 'operation_description', 'operation_result', 'uid'],
    ['event_id', 'operation_description', 'operation_type', 'uid'],
    ['event_id', 'operation_type', 'source_ip', 'tenant_id'],
    ['event_id', 'operation_result', 'source_ip', 'tenant_id'],
    ['event_id', 'operation_description', 'source_ip', 'tenant_id'],
    ['event_id', 'operation_type', 'source_ip', 'user_agent'],
    ['event_id', 'operation_result', 'source_ip', 'user_agent'],
    ['event_id', 'operation_description', 'source_ip', 'user_agent'],
    ['event_id', 'operation_result', 'source_ip', 'uid'],
    ['event_id', 'operation_description', 'source_ip', 'uid'],
    ['event_id', 'operation_type', 'source_ip', 'uid'],
    ['event_id', 'operation_type', 'tenant_id', 'user_agent'],
    ['event_id', 'operation_result', 'tenant_id', 'user_agent'],
    ['event_id', 'operation_description', 'tenant_id', 'user_agent'],
    ['event_id', 'operation_type', 'tenant_id', 'uid'],
    ['event_id', 'operation_result', 'tenant_id', 'uid'],
    ['event_id', 'operation_description', 'tenant_id', 'uid'],
    ['event_id', 'operation_type', 'uid', 'user_agent'],
    ['event_id', 'operation_result', 'uid', 'user_agent'],
    ['event_id', 'operation_description', 'uid', 'user_agent'],
    ['event_id', 'source_ip', 'tenant_id', 'user_agent'],
    ['event_id', 'source_ip', 'tenant_id', 'uid'],
    ['event_id', 'source_ip', 'uid', 'user_agent'],
    ['event_id', 'tenant_id', 'uid', 'user_agent'],
    ['operation_description', 'operation_result', 'operation_type', 'source_ip'],
    ['operation_description', 'operation_result', 'operation_type', 'tenant_id'],
    ['operation_description', 'operation_result', 'operation_type', 'user_agent'],
    ['operation_description', 'operation_result', 'operation_type', 'uid'],
    ['operation_result', 'operation_type', 'source_ip', 'tenant_id'],
    ['operation_description', 'operation_result', 'source_ip', 'tenant_id'],
    ['operation_description', 'operation_type', 'source_ip', 'tenant_id'],
    ['operation_result', 'operation_type', 'source_ip', 'user_agent'],
    ['operation_description', 'operation_result', 'source_ip', 'user_agent'],
    ['operation_description', 'operation_type', 'source_ip', 'user_agent'],
    ['operation_description', 'operation_result', 'source_ip', 'uid'],
    ['operation_result', 'operation_type', 'source_ip', 'uid'],
    ['operation_description', 'operation_type', 'source_ip', 'uid'],
    ['operation_result', 'operation_type', 'tenant_id', 'user_agent'],
    ['operation_description', 'operation_result', 'tenant_id', 'user_agent'],
    ['operation_description', 'operation_type', 'tenant_id', 'user_agent'],
    ['operation_result', 'operation_type', 'tenant_id', 'uid'],
    ['operation_description', 'operation_result', 'tenant_id', 'uid'],
    ['operation_description', 'operation_type', 'tenant_id', 'uid'],
    ['operation_result', 'operation_type', 'uid', 'user_agent'],
    ['operation_description', 'operation_result', 'uid', 'user_agent'],
    ['operation_description', 'operation_type', 'uid', 'user_agent'],
    ['operation_result', 'source_ip', 'tenant_id', 'user_agent'],
    ['operation_description', 'source_ip', 'tenant_id', 'user_agent'],
    ['operation_type', 'source_ip', 'tenant_id', 'user_agent'],
    ['operation_result', 'source_ip', 'tenant_id', 'uid'],
    ['operation_description', 'source_ip', 'tenant_id', 'uid'],
    ['operation_type', 'source_ip', 'tenant_id', 'uid'],
    ['operation_result', 'source_ip', 'uid', 'user_agent'],
    ['operation_description', 'source_ip', 'uid', 'user_agent'],
    ['operation_type', 'source_ip', 'uid', 'user_agent'],
    ['operation_result', 'tenant_id', 'uid', 'user_agent'],
    ['operation_description', 'tenant_id', 'uid', 'user_agent'],
    ['operation_type', 'tenant_id', 'uid', 'user_agent'],
    ['source_ip', 'tenant_id', 'uid', 'user_agent']],

    '5 crosses': [
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'operation_type'],
    ['app_id', 'event_id', 'operation_result', 'operation_type', 'source_ip'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'source_ip'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'source_ip'],
    ['app_id', 'event_id', 'operation_result', 'operation_type', 'tenant_id'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'tenant_id'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'tenant_id'],
    ['app_id', 'event_id', 'operation_result', 'operation_type', 'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'operation_type', 'uid'],
    ['app_id', 'event_id', 'operation_description', 'operation_result', 'uid'],
    ['app_id', 'event_id', 'operation_description', 'operation_type', 'uid'],
    ['app_id', 'event_id', 'operation_type', 'source_ip', 'tenant_id'],
    ['app_id', 'event_id', 'operation_result', 'source_ip', 'tenant_id'],
    ['app_id', 'event_id', 'operation_description', 'source_ip', 'tenant_id'],
    ['app_id', 'event_id', 'operation_type', 'source_ip', 'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'source_ip', 'user_agent'],
    ['app_id', 'event_id', 'operation_description', 'source_ip', 'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'source_ip', 'uid'],
    ['app_id', 'event_id', 'operation_description', 'source_ip', 'uid'],
    ['app_id', 'event_id', 'operation_type', 'source_ip', 'uid'],
    ['app_id', 'event_id', 'operation_type', 'tenant_id', 'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'tenant_id', 'user_agent'],
    ['app_id', 'event_id', 'operation_description', 'tenant_id', 'user_agent'],
    ['app_id', 'event_id', 'operation_type', 'tenant_id', 'uid'],
    ['app_id', 'event_id', 'operation_result', 'tenant_id', 'uid'],
    ['app_id', 'event_id', 'operation_description', 'tenant_id', 'uid'],
    ['app_id', 'event_id', 'operation_type', 'uid', 'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'uid', 'user_agent'],
    ['app_id', 'event_id', 'operation_description', 'uid', 'user_agent'],
    ['app_id', 'event_id', 'source_ip', 'tenant_id', 'user_agent'],
    ['app_id', 'event_id', 'source_ip', 'tenant_id', 'uid'],
    ['app_id', 'event_id', 'source_ip', 'uid', 'user_agent'],
    ['app_id', 'event_id', 'tenant_id', 'uid', 'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'uid'],
    ['app_id', 'operation_result', 'operation_type', 'source_ip', 'tenant_id'],
    ['app_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id'],
    ['app_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['app_id', 'operation_result', 'operation_type', 'source_ip', 'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['app_id', 'operation_description', 'operation_result', 'source_ip', 'uid'],
    ['app_id', 'operation_result', 'operation_type', 'source_ip', 'uid'],
    ['app_id', 'operation_description', 'operation_type', 'source_ip', 'uid'],
    ['app_id', 'operation_result', 'operation_type', 'tenant_id', 'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['app_id', 'operation_result', 'operation_type', 'tenant_id', 'uid'],
    ['app_id', 'operation_description', 'operation_result', 'tenant_id', 'uid'],
    ['app_id', 'operation_description', 'operation_type', 'tenant_id', 'uid'],
    ['app_id', 'operation_result', 'operation_type', 'uid', 'user_agent'],
    ['app_id', 'operation_description', 'operation_result', 'uid', 'user_agent'],
    ['app_id', 'operation_description', 'operation_type', 'uid', 'user_agent'],
    ['app_id', 'operation_result', 'source_ip', 'tenant_id', 'user_agent'],
    ['app_id', 'operation_description', 'source_ip', 'tenant_id', 'user_agent'],
    ['app_id', 'operation_type', 'source_ip', 'tenant_id', 'user_agent'],
    ['app_id', 'operation_result', 'source_ip', 'tenant_id', 'uid'],
    ['app_id', 'operation_description', 'source_ip', 'tenant_id', 'uid'],
    ['app_id', 'operation_type', 'source_ip', 'tenant_id', 'uid'],
    ['app_id', 'operation_result', 'source_ip', 'uid', 'user_agent'],
    ['app_id', 'operation_description', 'source_ip', 'uid', 'user_agent'],
    ['app_id', 'operation_type', 'source_ip', 'uid', 'user_agent'],
    ['app_id', 'operation_result', 'tenant_id', 'uid', 'user_agent'],
    ['app_id', 'operation_description', 'tenant_id', 'uid', 'user_agent'],
    ['app_id', 'operation_type', 'tenant_id', 'uid', 'user_agent'],
    ['app_id', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'uid'],
    ['event_id', 'operation_result', 'operation_type', 'source_ip', 'tenant_id'],
    ['event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id'],
    ['event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['event_id', 'operation_result', 'operation_type', 'source_ip', 'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['event_id', 'operation_description', 'operation_result', 'source_ip', 'uid'],
    ['event_id', 'operation_result', 'operation_type', 'source_ip', 'uid'],
    ['event_id', 'operation_description', 'operation_type', 'source_ip', 'uid'],
    ['event_id', 'operation_result', 'operation_type', 'tenant_id', 'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'tenant_id',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['event_id', 'operation_result', 'operation_type', 'tenant_id', 'uid'],
    ['event_id', 'operation_description', 'operation_result', 'tenant_id', 'uid'],
    ['event_id', 'operation_description', 'operation_type', 'tenant_id', 'uid'],
    ['event_id', 'operation_result', 'operation_type', 'uid', 'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'uid',
    'user_agent'],
    ['event_id', 'operation_description', 'operation_type', 'uid', 'user_agent'],
    ['event_id', 'operation_result', 'source_ip', 'tenant_id', 'user_agent'],
    ['event_id', 'operation_description', 'source_ip', 'tenant_id', 'user_agent'],
    ['event_id', 'operation_type', 'source_ip', 'tenant_id', 'user_agent'],
    ['event_id', 'operation_result', 'source_ip', 'tenant_id', 'uid'],
    ['event_id', 'operation_description', 'source_ip', 'tenant_id', 'uid'],
    ['event_id', 'operation_type', 'source_ip', 'tenant_id', 'uid'],
    ['event_id', 'operation_result', 'source_ip', 'uid', 'user_agent'],
    ['event_id', 'operation_description', 'source_ip', 'uid', 'user_agent'],
    ['event_id', 'operation_type', 'source_ip', 'uid', 'user_agent'],
    ['event_id', 'operation_result', 'tenant_id', 'uid', 'user_agent'],
    ['event_id', 'operation_description', 'tenant_id', 'uid', 'user_agent'],
    ['event_id', 'operation_type', 'tenant_id', 'uid', 'user_agent'],
    ['event_id', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'uid',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['operation_description',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'uid'],
    ['operation_result', 'operation_type', 'source_ip', 'tenant_id', 'uid'],
    ['operation_description', 'operation_type', 'source_ip', 'tenant_id', 'uid'],
    ['operation_description',
    'operation_result',
    'source_ip',
    'uid',
    'user_agent'],
    ['operation_result', 'operation_type', 'source_ip', 'uid', 'user_agent'],
    ['operation_description', 'operation_type', 'source_ip', 'uid', 'user_agent'],
    ['operation_description',
    'operation_result',
    'tenant_id',
    'uid',
    'user_agent'],
    ['operation_result', 'operation_type', 'tenant_id', 'uid', 'user_agent'],
    ['operation_description', 'operation_type', 'tenant_id', 'uid', 'user_agent'],
    ['operation_result', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['operation_description', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['operation_type', 'source_ip', 'tenant_id', 'uid', 'user_agent']],

    '6 crosses': [
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'uid'],
    ['app_id',
    'event_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['app_id',
    'event_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'uid'],
    ['app_id',
    'event_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'uid'],
    ['app_id',
    'event_id',
    'operation_result',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'tenant_id',
    'uid'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'tenant_id',
    'uid'],
    ['app_id',
    'event_id',
    'operation_result',
    'operation_type',
    'uid',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_result',
    'uid',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'operation_type',
    'uid',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_result',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'event_id',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'source_ip', 'tenant_id', 'uid'],
    ['app_id',
    'event_id',
    'operation_description',
    'source_ip',
    'tenant_id',
    'uid'],
    ['app_id', 'event_id', 'operation_type', 'source_ip', 'tenant_id', 'uid'],
    ['app_id', 'event_id', 'operation_result', 'source_ip', 'uid', 'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'source_ip',
    'uid',
    'user_agent'],
    ['app_id', 'event_id', 'operation_type', 'source_ip', 'uid', 'user_agent'],
    ['app_id', 'event_id', 'operation_result', 'tenant_id', 'uid', 'user_agent'],
    ['app_id',
    'event_id',
    'operation_description',
    'tenant_id',
    'uid',
    'user_agent'],
    ['app_id', 'event_id', 'operation_type', 'tenant_id', 'uid', 'user_agent'],
    ['app_id', 'event_id', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid'],
    ['app_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'uid',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'uid'],
    ['app_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid'],
    ['app_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid'],
    ['app_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'uid',
    'user_agent'],
    ['app_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'uid',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_result',
    'tenant_id',
    'uid',
    'user_agent'],
    ['app_id',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid',
    'user_agent'],
    ['app_id',
    'operation_description',
    'operation_type',
    'tenant_id',
    'uid',
    'user_agent'],
    ['app_id', 'operation_result', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['app_id',
    'operation_description',
    'source_ip',
    'tenant_id',
    'uid',
    'user_agent'],
    ['app_id', 'operation_type', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid'],
    ['event_id',
    'operation_description',
    'operation_result',
    'operation_type',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['event_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'uid'],
    ['event_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid'],
    ['event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid'],
    ['event_id',
    'operation_description',
    'operation_result',
    'source_ip',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_type',
    'source_ip',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_result',
    'tenant_id',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_description',
    'operation_type',
    'tenant_id',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_result',
    'source_ip',
    'tenant_id',
    'uid',
    'user_agent'],
    ['event_id',
    'operation_description',
    'source_ip',
    'tenant_id',
    'uid',
    'user_agent'],
    ['event_id', 'operation_type', 'source_ip', 'tenant_id', 'uid', 'user_agent'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'source_ip',
    'uid',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'operation_type',
    'tenant_id',
    'uid',
    'user_agent'],
    ['operation_description',
    'operation_result',
    'source_ip',
    'tenant_id',
    'uid',
    'user_agent'],
    ['operation_description',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid',
    'user_agent'],
    ['operation_result',
    'operation_type',
    'source_ip',
    'tenant_id',
    'uid',
    'user_agent']
    ]
}
