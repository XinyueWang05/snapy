"""Require the two ranks to fail at Gloo's requested interface lookup."""

import subprocess
import sys


def main():
    try:
        result = subprocess.run(
            sys.argv[1:],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Gloo interface test could not complete: {exc}")
        return 1

    expected = "Unable to find address for: snapy_no_such_interface"
    if result.returncode != 0 and result.stdout.count(expected) == 2:
        return 0

    print(result.stdout)
    print(
        f"Expected two Gloo interface errors; got return code "
        f"{result.returncode} and {result.stdout.count(expected)} matches"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
