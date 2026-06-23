"""
Tests for HTCondor environment variable handling.

Covers:
1. _format_htcondor_environment() serialization to HTCondor new-style format
2. Verification that CommonSettings flags are configured correctly for the
   HTCondor executor's environment injection strategy.
"""

from unittest.mock import Mock
from snakemake_executor_plugin_htcondor import Executor, common_settings


# ---------------------------------------------------------------------------
# 1. _format_htcondor_environment() serialization
# ---------------------------------------------------------------------------


class TestFormatHtcondorEnvironment:
    """Tests for _format_htcondor_environment() serialization."""

    def setup_method(self):
        self.executor = Mock(spec=Executor)
        self.executor._format_htcondor_environment = (
            Executor._format_htcondor_environment.__get__(self.executor, Executor)
        )

    # The method returns an HTCondor new-syntax environment *body* (the
    # NAME=value assignments that go inside the outer double quotes, which the
    # caller adds).  A simple value is emitted bare; only values containing
    # whitespace are single-quoted.  Literal double/single quotes are doubled.

    def test_single_variable(self):
        """Single key-value pair is serialized as a bare NAME=value assignment."""
        result = self.executor._format_htcondor_environment({"FOO": "bar"})
        assert result == "FOO=bar"

    def test_multiple_variables(self):
        """Multiple variables are space-separated NAME=value assignments."""
        result = self.executor._format_htcondor_environment(
            {"A": "1", "B": "2", "C": "3"}
        )
        assert "A=1" in result
        assert "B=2" in result
        assert "C=3" in result
        # They should be space-separated
        parts = result.split(" ")
        assert len(parts) == 3

    def test_empty_dict(self):
        """An empty dict produces an empty string."""
        result = self.executor._format_htcondor_environment({})
        assert result == ""

    def test_value_with_double_quotes_escaped(self):
        """Double quotes inside values are doubled; the spaced value is single-quoted."""
        result = self.executor._format_htcondor_environment({"MSG": 'say "hello"'})
        assert result == 'MSG=\'say ""hello""\''

    def test_value_with_double_quotes_no_space(self):
        """Embedded double quotes are doubled even without surrounding spaces."""
        result = self.executor._format_htcondor_environment({"V": 'pre"mid"post'})
        assert result == 'V=pre""mid""post'

    def test_value_with_spaces(self):
        """Values with spaces are wrapped in single quotes."""
        result = self.executor._format_htcondor_environment(
            {"PATHLIST": "/usr/bin /usr/local/bin"}
        )
        assert result == "PATHLIST='/usr/bin /usr/local/bin'"

    def test_value_with_single_quote_escaped(self):
        """Single quotes inside a (spaced) value are doubled."""
        result = self.executor._format_htcondor_environment({"MSG": "it's here"})
        assert result == "MSG='it''s here'"

    def test_value_with_equals_sign(self):
        """Values containing = need no special quoting when there is no space."""
        result = self.executor._format_htcondor_environment({"OPTS": "key=value"})
        assert result == "OPTS=key=value"

    def test_non_string_value_coerced(self):
        """Non-string values are coerced to strings."""
        result = self.executor._format_htcondor_environment({"COUNT": 42, "FLAG": True})
        assert "COUNT=42" in result
        assert "FLAG=True" in result


# ---------------------------------------------------------------------------
# 2. CommonSettings flag correctness
# ---------------------------------------------------------------------------


class TestCommonSettingsFlags:
    """Verify that CommonSettings flags are set correctly for the HTCondor executor."""

    def test_pass_envvar_declarations_to_cmd_is_false(self):
        """pass_envvar_declarations_to_cmd must be False.

        When True, Snakemake embeds "export VAR=val &&" at the start of the
        arguments string.  This breaks the job_wrapper code path because the
        arguments string must start cleanly with "python -m snakemake" for
        prefix-stripping to succeed.  Instead, the executor injects env vars
        via HTCondor's native ``environment`` key in run_job().
        """
        assert common_settings.pass_envvar_declarations_to_cmd is False

    def test_job_deploy_sources_is_false(self):
        """job_deploy_sources must be False.

        This setting only takes effect when can_transfer_local_files is False.
        Since the HTCondor executor sets can_transfer_local_files=True (it handles
        its own file transfers), job_deploy_sources is a dead letter.  Setting it
        to False avoids confusion.
        """
        assert common_settings.job_deploy_sources is False

    def test_can_transfer_local_files_is_true(self):
        """can_transfer_local_files must be True.

        This tells Snakemake that the executor handles its own file transfers
        (via HTCondor's transfer_input_files / transfer_output_files), so
        Snakemake should not require a default storage provider for local files.
        """
        assert common_settings.can_transfer_local_files is True

    def test_non_local_exec_is_true(self):
        """non_local_exec must be True — jobs run on remote execution points."""
        assert common_settings.non_local_exec is True

    def test_pass_default_storage_provider_args_is_true(self):
        """pass_default_storage_provider_args must be True.

        EP jobs need the --default-storage-provider and --default-storage-prefix
        arguments so that storage-backed I/O works correctly on the remote side.
        """
        assert common_settings.pass_default_storage_provider_args is True

    def test_pass_default_resources_args_is_true(self):
        """pass_default_resources_args must be True.

        EP jobs need the --default-resources arguments so that resource defaults
        (e.g. tmpdir=$_CONDOR_SCRATCH_DIR) are propagated to remote execution.
        """
        assert common_settings.pass_default_resources_args is True
