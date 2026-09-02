#!/usr/bin/env python3
"""The README tells people which version to depend on, and in tflite_beam that
line went three releases stale without anything noticing. The same line exists
here, written by hand next to a paragraph that describes what the version
contains, so it goes stale the same way."""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

MIX_EXS = ROOT / "mix.exs"
README = ROOT / "README.md"


def project_version() -> str:
    match = re.search(r'@version\s+"([^"]+)"', MIX_EXS.read_text())
    if not match:
        sys.exit(f"no @version in {MIX_EXS}")
    return match.group(1)


def readme_requirements() -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(README.read_text().splitlines(), 1):
        match = re.search(r'\{:tflite_elixir,\s*"([^"]+)"\}', line)
        if match:
            found.append((number, match.group(1)))
    return found


def is_exact(requirement: str) -> bool:
    """A range such as ~> 0.3 names a family and does not go stale; an exact
    version names one release and does."""
    return not requirement.strip().startswith(("~>", ">=", "<=", ">", "<", "=="))


def main() -> int:
    version = project_version()
    prerelease = "-" in version
    found = readme_requirements()
    if not found:
        print('no {:tflite_elixir, "..."} requirement in README.md to check')
        return 0

    stale = []
    for number, requirement in found:
        if not is_exact(requirement) or requirement == version:
            continue
        # A pre-release has to be named exactly, so the README carries one and it
        # is the one that goes stale. A requirement that is itself a release is a
        # deliberate reference to an older version.
        if "-" in requirement or (prerelease and requirement.startswith(version.split("-")[0])):
            stale.append((number, requirement))

    print(f"mix.exs: {version}")
    for number, requirement in found:
        mark = "stale" if (number, requirement) in stale else "ok"
        print(f'  README.md:{number}: {{:tflite_elixir, "{requirement}"}}  [{mark}]')

    if stale:
        print()
        for number, requirement in stale:
            print(f"README.md:{number} tells people to depend on {requirement}, "
                  f"but this is {version}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
