# import config.from_args before all other internals or never
from . import Config
Config.from_args()
