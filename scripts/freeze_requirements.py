"""Lock the installed transitive closure of declared project/dev dependencies."""

from importlib.metadata import distribution
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def main():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pending = list(
        config["project"]["dependencies"] + config["project"]["optional-dependencies"]["dev"]
    )
    pinned = {}
    marker_env = {**default_environment(), "extra": ""}
    while pending:
        requirement = Requirement(pending.pop())
        if requirement.marker and not requirement.marker.evaluate(marker_env):
            continue
        key = canonicalize_name(requirement.name)
        if key in pinned:
            continue
        package = distribution(requirement.name)
        if requirement.specifier and package.version not in requirement.specifier:
            raise RuntimeError(f"Installed {key}=={package.version} violates {requirement}")
        pinned[key] = package.version
        pending.extend(package.requires or [])
    # Platform-only indirect packages must not make the lock fail on CI Linux.
    platform_specific = {
        "pywin32": "sys_platform == 'win32'",
        "colorama": "sys_platform == 'win32'",
    }
    lines = ["# Python 3.12; tested dependency closure. Regenerate only after validating tests."]
    for name, number in sorted(pinned.items()):
        suffix = f"; {platform_specific[name]}" if name in platform_specific else ""
        lines.append(f"{name}=={number}{suffix}")
    (ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Locked {len(pinned)} distributions.")


if __name__ == "__main__":
    main()
