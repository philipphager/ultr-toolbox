import functools
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Union

logger = logging.getLogger(__name__)


def cache(
        directory: Union[str, Path],
) -> Any:
    def wrapper_factory(func: Callable):
        base_dir = Path(directory).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)

        @functools.wraps(func)
        def wrapper(*args):
            key = hashlib.sha256()

            for arg in args:
                key.update(str.encode(str(arg)))

            key = key.hexdigest()

            file = Path(f"{key}.pickle")
            path = (base_dir / file).resolve()

            if not path.exists():
                output = func(*args)
                pickle.dump(output, open(path, "wb"))
                print(f"Storing output of '{func.__name__}{args}' to: {path}")
                return output
            else:
                print(f"Loading output of '{func.__name__}{args}' from: {path}")
                return pickle.load(open(path, "rb"))

        return wrapper

    return wrapper_factory
