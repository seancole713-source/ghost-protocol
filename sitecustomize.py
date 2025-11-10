# Ensure third-party pytest plugins are not auto-loaded during test runs
# This avoids import errors from unrelated dependencies providing pytest11 entry points.
import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
