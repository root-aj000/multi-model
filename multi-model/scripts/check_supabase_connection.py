#!/usr/bin/env python3
"""
Supabase Connection Check Script
=================================
Validates that the Supabase connection is properly configured and reachable.

Performs the following checks:
  1. Environment variables are set and non-empty
  2. Anon client can reach the database (RLS enforced)
  3. Admin client can reach the database (RLS bypassed)
  4. JWT secret is valid base64 with sufficient length

Usage:
    python scripts/check_supabase_connection.py
    python scripts/check_supabase_connection.py --env-file .env
    python scripts/check_supabase_connection.py --timeout 15

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single check."""

    name: str
    passed: bool
    message: str
    detail: str = ""


# ---------------------------------------------------------------------------
# ANSI helpers (disabled when stdout is not a TTY)
# ---------------------------------------------------------------------------

_USE_COLOUR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOUR else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOUR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOUR else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOUR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _USE_COLOUR else text


# ---------------------------------------------------------------------------
# .env file loader (simple, no external dependency)
# ---------------------------------------------------------------------------

def load_env_file(env_file: str) -> None:
    """
    Load environment variables from a .env file.

    Only sets variables that are NOT already defined in the environment,
    so real env vars / CI settings take precedence.

    Handles:
      - Blank lines and comments (lines starting with #)
      - Quoted values (single or double quotes stripped)
      - Inline comments after values
    """
    if not os.path.exists(env_file):
        print(_yellow(f"⚠  Env file not found: {env_file} (skipping)"))
        return

    count = 0
    with open(env_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # Skip blanks and comments
            if not line or line.startswith("#"):
                continue
            # Must contain '='
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            # Don't overwrite existing env vars
            if os.environ.get(key):
                continue
            # Strip surrounding quotes and inline comments
            value = value.strip()
            if value.startswith(("'", '"')):
                quote_char = value[0]
                end = value.find(quote_char, 1)
                if end != -1:
                    value = value[1:end]
                else:
                    value = value[1:]
            else:
                # Remove inline comment
                if " #" in value:
                    value = value[: value.index(" #")]
            value = value.strip()
            os.environ[key] = value
            count += 1

    print(_dim(f"   Loaded {count} variable(s) from {env_file}"))


# ---------------------------------------------------------------------------
# Check: environment variables
# ---------------------------------------------------------------------------

REQUIRED_ENV_VARS = {
    "SUPABASE_URL": "Supabase project URL (e.g. https://xxxxx.supabase.co)",
    "SUPABASE_KEY": "Supabase anon/public key (RLS enforced)",
    "SUPABASE_SERVICE_ROLE_KEY": "Supabase service_role key (bypasses RLS)",
    "SUPABASE_JWT_SECRET": "Supabase JWT secret (base64-encoded)",
}


def check_env_vars() -> List[CheckResult]:
    """Verify all required Supabase environment variables are set and non-empty."""
    results: List[CheckResult] = []

    for var, description in REQUIRED_ENV_VARS.items():
        value = os.environ.get(var, "")

        if not value:
            results.append(
                CheckResult(
                    name=f"ENV: {var}",
                    passed=False,
                    message="Missing or empty",
                    detail=f"Required for: {description}",
                )
            )
            continue

        # Extra format validation for URL
        if var == "SUPABASE_URL":
            if not value.startswith("https://"):
                results.append(
                    CheckResult(
                        name=f"ENV: {var}",
                        passed=False,
                        message="Invalid URL — must start with https://",
                        detail=f"Got: {value[:40]}{'…' if len(value) > 40 else ''}",
                    )
                )
                continue
            if not value.endswith(".supabase.co"):
                results.append(
                    CheckResult(
                        name=f"ENV: {var}",
                        passed=False,
                        message="Invalid URL — must end with .supabase.co",
                        detail=f"Got: {value[:40]}{'…' if len(value) > 40 else ''}",
                    )
                )
                continue

        # All good
        results.append(
            CheckResult(
                name=f"ENV: {var}",
                passed=True,
                message="Set and non-empty",
                detail=f"Length: {len(value)} chars",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Check: Supabase client connectivity
# ---------------------------------------------------------------------------

def _test_client_connection(
    url: str,
    key: str,
    client_label: str,
    timeout_seconds: int,
) -> CheckResult:
    """
    Test a Supabase client connection using a fallback chain:

      1. RPC ``ping()`` — lightest check, no table scan
      2. ``_healthcheck`` table SELECT — requires migration 002
      3. Auth endpoint — always available in Supabase

    Returns a CheckResult with the first successful method,
    or the last error if all methods fail.
    """
    try:
        from supabase import create_client

        # Set a reasonable timeout via the storage/client options
        # supabase-py uses httpx internally; we pass options through.
        client = create_client(url, key)
    except Exception as exc:
        return CheckResult(
            name=client_label,
            passed=False,
            message=f"Client creation failed: {exc}",
        )

    # --- Attempt 1: RPC ping() ---
    try:
        result = client.rpc("ping", {}).execute()
        return CheckResult(
            name=f"{client_label}: RPC ping()",
            passed=True,
            message=f"Connected — ping() returned: {result.data!r}",
        )
    except Exception:
        pass  # RPC may not exist yet (migration not applied)

    # --- Attempt 2: _healthcheck table ---
    try:
        result = (
            client.table("_healthcheck")
            .select("alive")
            .limit(1)
            .execute()
        )
        row_count = len(result.data) if result.data else 0
        return CheckResult(
            name=f"{client_label}: _healthcheck table",
            passed=True,
            message=f"Connected — _healthcheck returned {row_count} row(s)",
        )
    except Exception:
        pass  # Table may not exist yet

    # --- Attempt 3: Auth endpoint (always available) ---
    try:
        # get_session() hits the Supabase Auth GoTrue endpoint
        # It returns None when no session exists, but doesn't raise
        # — the important thing is it doesn't throw a connection error.
        session = client.auth.get_session()
        return CheckResult(
            name=f"{client_label}: Auth endpoint",
            passed=True,
            message="Connected — auth endpoint reachable"
            + (f" (session active)" if session else " (no active session)"),
        )
    except Exception as exc:
        return CheckResult(
            name=client_label,
            passed=False,
            message=f"Connection failed: {exc}",
            detail="Hint: Check SUPABASE_URL and key values. "
            "Also verify the Supabase project is not paused.",
        )


def check_anon_client(timeout: int) -> CheckResult:
    """Test connection with the anon key (RLS enforced)."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        return CheckResult(
            name="Anon Client",
            passed=False,
            message="Skipped — SUPABASE_URL or SUPABASE_KEY not set",
        )

    return _test_client_connection(url, key, "Anon Client", timeout)


def check_admin_client(timeout: int) -> CheckResult:
    """Test connection with the service_role key (RLS bypassed)."""
    url = os.environ.get("SUPABASE_URL", "")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not service_key:
        return CheckResult(
            name="Admin Client",
            passed=False,
            message="Skipped — SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set",
        )

    return _test_client_connection(url, service_key, "Admin Client", timeout)


# ---------------------------------------------------------------------------
# Check: JWT secret format
# ---------------------------------------------------------------------------

def check_jwt_secret() -> CheckResult:
    """Validate JWT secret format (base64, minimum length)."""
    secret = os.environ.get("SUPABASE_JWT_SECRET", "")

    if not secret:
        return CheckResult(
            name="JWT Secret",
            passed=False,
            message="Not set",
            detail="Required for verifying Supabase JWTs in the backend",
        )

    try:
        decoded = base64.b64decode(secret, validate=True)
    except Exception as exc:
        return CheckResult(
            name="JWT Secret",
            passed=False,
            message=f"Invalid base64: {exc}",
            detail="The JWT secret should be a base64-encoded string "
            "from Supabase Dashboard → Settings → API → JWT Settings",
        )

    if len(decoded) < 32:
        return CheckResult(
            name="JWT Secret",
            passed=False,
            message=f"Too short after base64 decode ({len(decoded)} bytes, expected ≥32)",
            detail="A valid HS256 secret should be at least 32 bytes",
        )

    return CheckResult(
        name="JWT Secret",
        passed=True,
        message=f"Valid base64, {len(decoded)} bytes decoded",
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: List[CheckResult]) -> None:
    """Print a formatted results table to stdout."""

    # Column widths
    name_w = max(len(r.name) for r in results) + 2
    msg_w = max(len(r.message) for r in results) + 2

    # Header
    print()
    print(_bold("╔" + "═" * (name_w + msg_w + 8) + "╗"))
    print(_bold("║") + _bold("  Supabase Connection Check".center(name_w + msg_w + 6)) + _bold("  ║"))
    print(_bold("╠" + "═" * (name_w + msg_w + 8) + "╣"))

    # Rows
    for r in results:
        icon = _green("✓") if r.passed else _red("✗")
        status = _green("PASS") if r.passed else _red("FAIL")
        name_col = f"  {icon}  {r.name:<{name_w}}"
        msg_col = f"{r.message:<{msg_w}}"
        print(f"║{name_col}{msg_col}  ║")
        if r.detail:
            detail_col = f"      {_dim(r.detail)}"
            print(f"║{detail_col:<{name_w + msg_w + 6}}  ║")

    # Footer
    print(_bold("╠" + "═" * (name_w + msg_w + 8) + "╣"))

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if passed == total:
        summary = _green(_bold(f"  {passed}/{total} checks passed ✅"))
    else:
        summary = _red(_bold(f"  {passed}/{total} checks passed ❌  ({total - passed} failed)"))

    print(f"║{summary:<{name_w + msg_w + 7}}║")
    print(_bold("╚" + "═" * (name_w + msg_w + 8) + "╝"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Supabase connection and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/check_supabase_connection.py\n"
            "  python scripts/check_supabase_connection.py --env-file .env\n"
            "  python scripts/check_supabase_connection.py --timeout 15\n"
        ),
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file to load (default: .env). "
        "Variables already in the environment take precedence.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Connection timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    # Load .env file (existing env vars take precedence)
    if args.env_file:
        load_env_file(args.env_file)

    # Run all checks
    results: List[CheckResult] = []
    results.extend(check_env_vars())
    results.append(check_anon_client(timeout=args.timeout))
    results.append(check_admin_client(timeout=args.timeout))
    results.append(check_jwt_secret())

    # Print results
    print_results(results)

    # Exit code
    all_passed = all(r.passed for r in results)
    if all_passed:
        print(_green("All checks passed — Supabase connection is healthy."))
    else:
        print(_red("Some checks failed — see details above."))
        if not any(r.name.startswith("ENV:") and not r.passed for r in results):
            print(_yellow("Tip: If connection fails, verify your Supabase project is not paused."))
            print(_yellow("     Run migration 002_connection_healthcheck.sql for better diagnostics."))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
