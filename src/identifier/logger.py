import logging
import sys
import os

from constants import LOG_DIR


log_fn = os.path.join(LOG_DIR, 'VideoSorting.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S',
                    filename=log_fn)
logger = logging.getLogger('VS')


def __log_init():
    if os.path.isfile(log_fn):
        os.remove(log_fn)


def __log_msg(msg):
    sys.stdout.write(msg + '\n')
    if len(msg) > 0 and msg[0] == '\r':
        msg = msg[1:]
    return msg


def info(msg):
    logging.info(__log_msg(msg))


def warn(msg):
    logging.warning(__log_msg(msg))


def error(msg):
    logging.error(__log_msg(msg))
