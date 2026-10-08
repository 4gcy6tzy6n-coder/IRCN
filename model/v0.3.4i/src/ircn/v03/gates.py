"""Fail-closed execution gates for downstream IRCN v0.3 work packages."""
from dataclasses import dataclass
from pathlib import Path

PREREQUISITES = {
    'WP2': ('WP1',),
    'WP3': ('WP1', 'WP2'),
    'WP4': ('WP1', 'WP2', 'WP3'),
    'WP5': ('WP1', 'WP2', 'WP3'),
    'WP6': ('WP1', 'WP2', 'WP3', 'ALGORITHM_FROZEN'),
}

@dataclass(frozen=True)
class GateResult:
    work_package: str
    allowed: bool
    blockers: tuple[str, ...]


def decide(work_package: str, statuses: dict[str, str]) -> GateResult:
    if work_package not in PREREQUISITES:
        raise ValueError(f'No downstream gate defined for {work_package!r}')
    blockers=[]
    for prerequisite in PREREQUISITES[work_package]:
        if statuses.get(prerequisite) != 'PASS':
            blockers.append(f'{prerequisite} status is {statuses.get(prerequisite, "UNKNOWN")}, requires PASS')
    return GateResult(work_package, not blockers, tuple(blockers))


def load_current_statuses(repo_root: Path) -> dict[str, str]:
    """Read explicit machine-readable gate evidence; missing files fail closed."""
    import json
    statuses={}
    for wp in ('WP1','WP2','WP3','ALGORITHM_FROZEN'):
        if wp=='WP1': path=repo_root/'reports/WP1/run_metadata.json'
        elif wp=='ALGORITHM_FROZEN': path=repo_root/'reports/ALGORITHM_FROZEN/status.json'
        else: path=repo_root/f'reports/{wp}/status.json'
        if not path.exists():
            statuses[wp]='UNKNOWN'; continue
        try:
            data=json.loads(path.read_text())
            statuses[wp]=str(data['status'])
        except (OSError,ValueError,KeyError,TypeError):
            statuses[wp]='INVALID'
    return statuses


def evaluate(repo_root: Path, work_package: str) -> GateResult:
    return decide(work_package,load_current_statuses(repo_root))
