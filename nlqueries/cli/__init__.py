"""nlqueries.cli — command-line interface package."""

from nlqueries.warning_filters import quiet_google_python_version_notice

# Before nlqueries.cli.main, whose connectors import Google's libraries.
quiet_google_python_version_notice()
