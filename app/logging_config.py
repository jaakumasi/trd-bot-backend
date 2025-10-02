import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging configuration for the trading bot"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with current date
    log_filename = f"trading_bot_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'console': {
                'format': '%(asctime)s | %(levelname)-8s | %(message)s',
                'datefmt': '%H:%M:%S'
            },
            'trading': {
                'format': '🤖 %(asctime)s | %(levelname)-8s | %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'trading',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': logs_dir / log_filename,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'trading_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': logs_dir / f"trades_{datetime.now().strftime('%Y-%m-%d')}.log",
                'maxBytes': 5242880,  # 5MB
                'backupCount': 10,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            'app.services.trading_bot': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'trading_file'],
                'propagate': False
            },
            'app.services.binance_service': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app.services.ai_analyzer': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app.services.risk_manager': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app.services.websocket_manager': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create a special logger for trading metrics
    trading_logger = logging.getLogger('trading_metrics')
    trading_logger.setLevel(logging.INFO)
    
    # Add a special handler for trading metrics
    metrics_handler = logging.handlers.RotatingFileHandler(
        logs_dir / f"metrics_{datetime.now().strftime('%Y-%m-%d')}.log",
        maxBytes=5242880,  # 5MB
        backupCount=10,
        encoding='utf8'
    )
    metrics_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s | METRICS | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    trading_logger.addHandler(metrics_handler)
    
    print("🔧 Logging system initialized")
    print(f"📝 Logs will be saved to: {logs_dir.absolute()}")
    
    return trading_logger

def get_trading_metrics_logger():
    """Get the trading metrics logger"""
    return logging.getLogger('trading_metrics')