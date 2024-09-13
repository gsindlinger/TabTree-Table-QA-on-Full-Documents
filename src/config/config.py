from __future__ import annotations
import logging
import os
import tomllib
import argparse
from pathlib import Path
from typing import Union, NoReturn
from abc import ABC, ABCMeta
from collections.abc import Hashable
from dotenv import dotenv_values, load_dotenv

from distutils.util import strtobool


def _readonly(self, *args, **kwargs) -> NoReturn:
    """
    This function just ignores arguments and raises an error.

    Used in Config to implement it as a read-only dict subclass.
    Inspired by Marcin Raczyński, mtraceur: https://stackoverflow.com/a/31049908

    Args:
      *args:
      **kwargs:
    """
    raise RuntimeError("Config is read-only!")


ConfigValue = Union[Hashable, "ConfigValue"]


class ConfigMeta(ABC, ABCMeta):
    """
    Metaclass for Config

    Used to enable singleton-like usage.
    """

    def __getattr__(cls, name: str) -> ConfigValue:
        return cls[name]

    @classmethod
    def __class_getitem__(cls, name: str) -> ConfigValue:
        pass


class Config(dict[str, ConfigValue], metaclass=ConfigMeta):
    """Singleton-like Config class reading from global config.toml with some extras"""

    _path_config: Path = Path("config.toml")
    _path_help: Path = Path(__file__).parent / "help.toml"
    _config: Config | None = None
    _help: dict | None

    def __init__(self, data: dict, _help: dict | None = None) -> None:
        # caveat: the file is only accessed by the class, not by instances!
        # instances are only created for sub-dicts

        for key, value in data.items():

            item_help = _help.get(key) if _help else None
            if not isinstance(key, str):
                raise TypeError(f"Config keys must be strings! Found {type(key)}")
            match value:
                case dict():
                    data[key] = Config(value, item_help)
                case Hashable():
                    pass
                case list():
                    data[key] = tuple(value)
                case t:
                    raise TypeError(f"unsupported Config value type: {t}")
            self.__annotations__[key] = type(value).__name__
        super().__setattr__("_help", _help)
        super().__init__(data)

    @classmethod
    def _read(cls) -> Config:
        """Create Config by reading global config, help file and env file"""

        # First get env variables
        env_vars = cls._load_env()

        # Load help file
        with open(cls._path_help, "rb") as f:
            _help = cls(tomllib.load(f))

        # Load config file
        with open(cls._path_config, "rb") as f:
            data = tomllib.load(f)

        data.update(env_vars)
        c = cls(data, _help)

        return c

    @classmethod
    def _load_env(cls) -> None:
        """Load environment variables from .env file"""
        return {"env_variables": dotenv_values()}

    @classmethod
    def from_args(cls) -> None:
        """Read config from file and override from sys.args"""
        if cls._config is not None:
            # this method may only be called once (typically in the .from_args module)!
            _readonly(cls)
        cls._config = cls._read()
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        cls._config._add_args(parser)
        args = parser.parse_args()
        for key_path, value in args._get_kwargs():
            o = cls._config
            keys = key_path.split(".")
            last_key = keys.pop()
            for key in keys:
                o = o[key]
                assert isinstance(o, Config), type(o)
            old = o[last_key]
            if old != value:
                super(cls, o).__setitem__(last_key, value)

    @classmethod
    def __class_getitem__(cls, name: str) -> ConfigValue:
        if cls._config is None:
            # the file is read only once, and only as soon as data is accessed!
            cls._config = cls._read()
        return cls._config[name]

    def _add_args(
        self,
        parser: argparse.ArgumentParser | argparse._ArgumentGroup,
        self_key: str = "",
    ) -> None:
        """
        Add config entries with as arguments with default values to an ArgumentParser.

        Args:
          parser: argparse.ArgumentParser | argparse._ArgumentGroup:
          self_key: str:  (Default value = "")

        Returns:

        """
        for subkey, value in self.items():
            key = f"{self_key}.{subkey}" if self_key else subkey
            match value:
                case Config():
                    group = parser.add_argument_group(key)
                    value._add_args(group, key)
                case _:
                    t = type(value)
                    if t is bool:

                        def t(val):
                            """

                            Args:
                              val:

                            Returns:

                            """
                            return bool(strtobool(val))

                    parser.add_argument(
                        f"--{key.replace('_', '-')}",
                        default=value,
                        type=t,
                        help=self._help[subkey] if self._help is not None else None,
                    )

    def __getattr__(self, name: str) -> ConfigValue:
        return self[name]

    # read-only dict implementation inspired by
    # Marcin Raczyński, mtraceur: https://stackoverflow.com/a/31049908
    __setitem__ = _readonly
    __delitem__ = _readonly
    setdefault = _readonly
    pop = _readonly
    popitem = _readonly
    update = _readonly
    clear = _readonly
    __setattr__ = _readonly
    __delattr__ = _readonly
