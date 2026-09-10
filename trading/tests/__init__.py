"""Test suite for the WSJ headline trading algorithm.

The package logs at INFO/WARNING as a matter of course; silence it here so
test output shows test results rather than trading commentary.
"""

import logging

logging.getLogger("wsj_headline_trader").addHandler(logging.NullHandler())
logging.getLogger("wsj_headline_trader").propagate = False
