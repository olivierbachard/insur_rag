import json
from pathlib import Path

from evals.models import TestQuestion

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


def load_tests() -> list[TestQuestion]:
    """Load test questions from JSONL file."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    return tests
