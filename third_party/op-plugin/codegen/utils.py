from dataclasses import dataclass
import os
import sys
import traceback
import warnings


class PathManager:

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the user
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def check_directory_path_readable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.R_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def remove_path_safety(cls, path: str):
        if os.path.islink(path):
            raise RuntimeError(f"Invalid path is a soft chain: {path}")
        if os.path.exists(path):
            os.remove(path)

def get_torchgen_dir():
    # get path of torchgen, then get tags.yaml and native_functions.yaml
    try:
        import torchgen
        return os.path.dirname(os.path.realpath(torchgen.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)