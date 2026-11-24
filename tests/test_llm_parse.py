import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_experiments"))
import serialize as S  # noqa: E402


def test_parse_json_and_fallbacks():
    assert S.parse('{"decision": "default", "probability_default": 72}') == (1, 0.72)
    assert S.parse('Sure: {"decision": "no_default", "probability_default": 10} ok') == (0, 0.10)
    assert S.parse("I think no_default.") == (0, None)
    assert S.parse("I cannot help with that.") == (None, None)
    assert S.parse('{"decision": "above_50k", "probability_above_50k": 60}', task="income") == (1, 0.6)
