"""TODO(ruilongli): DO NOT SUBMIT without one-line documentation for test_visualizer.

TODO(ruilongli): DO NOT SUBMIT without a detailed description of test_visualizer.
"""

from typing import Sequence

from absl import app


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)