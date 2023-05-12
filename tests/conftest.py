import os

import pooch
from pooch import Untar


# Download data if necessary
def pytest_configure():
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/7916559/files/Data.tar.gz?download=1",
        processor=Untar(),
        known_hash="c27a566ed280f3a579dd2bf6846"
        "2a427d7d0d1286cdd8db2a4a035495f40f7e4",
    )
    symlink_args = dict(
        src=f"{os.path.commonpath(fnames)}",
        dst="./tests/Data",
        target_is_directory=True,
    )
    try:
        print(f"Linking {symlink_args['src']!r} to {symlink_args['dst']!r}")
        os.symlink(**symlink_args)
    except FileExistsError:
        os.unlink("./tests/Data")
        os.symlink(**symlink_args)
