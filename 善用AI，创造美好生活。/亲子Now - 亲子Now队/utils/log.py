"""
Copyright: Qinzi Now, Tencent Cloud.
"""

import logging

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s', level=logging.DEBUG)


class Log:

    def __init__(self, name):
        self._logger = logging.getLogger(name)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)