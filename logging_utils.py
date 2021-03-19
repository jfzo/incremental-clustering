import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1, # required
    'disable_existing_loggers': True, # this config overrides all other loggers
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s -- %(message)s'
        },
        'whenAndWhere': {
            'format': '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'whenAndWhere'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'whenAndWhere',
            'filename':'log/ilsh.log'

        }
    },
    'loggers': {
        'ilshsrc': { # 'root' logger
            'level': 'DEBUG', #CRITICAL will only print Errors
            'handlers': ['console','file']
        }
    }
}

def get_ilshlogger():
    """
    Gets the logger object for debugging purposes.
    :return: a logger
    """
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger('ilshsrc')