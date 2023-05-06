# run.py
"""
Build documentations and run ihLDA
    :code:`python run.py --mode a`
"""

import subprocess
import argparse
import sys


def run():
    """Run ihLDA

    This function runs Sphinx to build documentation, :code:`setup.py`
    to compile Cython, and :code:`main.py` to start the estimation.

    In terminal,

    >>> $ python run.py

    You can use command line arguments. :code:`--mode p` runs Python
    only while :code:`--mode s` runs Sphinx. If you do not
    set any arguments,
    or set :code:`--mode a`, we execute both.

    If you pass
    :code:`--mode g`, it will ask you to enter a GitHub commit comment.

    If you pass
    :code:`--mode profiler`, it will run with profiler.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="p",
                        help="select mode")
    args = parser.parse_args()

    if args.mode:
        if args.mode == "p":
            print("=== Run Python ==")
            commands = [["python", "setup.py", "build_ext", "--inplace"],
                        ["python", "main.py"]
                        ]  # Cython and main function

        elif args.mode == "O":
            print("=== Run Python skipping assert ==")
            commands = [["python", "setup.py", "build_ext", "--inplace"],
                        ["python", "-O", "main.py"]
                        ]  # Cython and main function


        elif args.mode == "profiler":
            print("=== Run with Profiler ===")
            commands = [
                    ["python", "setup.py", "build_ext", "--inplace"],
                    ["python", "-m", "cProfile", "-o", "profile.log",
                        "-s", "time", "main.py"],
                    ["cprofilev", "-f", "profile.log"]
                ]

        else:
            print("Option is wrong")
            sys.exit()

    for cmd in commands:
        subprocess.run(cmd)


if __name__ == '__main__':
    run()
