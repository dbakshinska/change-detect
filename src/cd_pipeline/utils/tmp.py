# src/cd_pipeline/utils/tmp.py
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path


def tmp_dir(prefix: str = "cd_", keep: bool = False) -> Path:
    """
    Create and return a temporary directory as a ``Path``.

    Parameters
    ----------
    prefix : str, default ``"cd_"``
        Prefix for the directory name (the final name is ``{prefix}_XXXXXX``).
    keep : bool, default ``False``
        If *True*, the directory is **not** deleted at interpreter exit.
        Stages that expose a ``--keep-temp`` flag should pass that value here.

    Returns
    -------
    pathlib.Path
        Path to the newly-created temporary directory.

    Notes
    -----
    * When *keep* is ``False`` (default) the directory is registered with
      :pymod:`atexit` for automatic recursive deletion.
    * You no longer need to use ``with tmp_dir():`` – just call it once:

        >>> work = tmp_dir("align")
        >>> (work / "foo.txt").write_text("hello")
    """
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    if not keep:
        atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path
