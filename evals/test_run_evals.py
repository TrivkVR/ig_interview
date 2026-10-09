import asyncio
import json
import types

from evals import run_evals as R


def test_dataset_shape():
    s = R.load_dataset()
    assert 15 <= len(s) <= 20
    assert {x["category"] for x in s} == {"safety_procedures", "maintenance_manuals", "quality_control_standards"}
    assert all(x["question"] and x["reference"] for x in s)


def test_mock_pipeline(tmp_path):
    out = tmp_path / "results.json"
    args = types.SimpleNamespace(mock=True, judge=False, limit=4, output=str(out))
    res = asyncio.run(R.main_async(args))
    data = json.loads(out.read_text())
    assert len(data["items"]) == 4 == len(res["items"])
    assert set(data["summary"]) == set(R.METRIC_NAMES)
    assert data["items"][0]["retrieved_contexts"]


def test_summarize_skips_nan_and_none():
    s = R.summarize([{"faithfulness": 1.0}, {"faithfulness": float("nan")}, {"faithfulness": 0.0}])
    assert s["faithfulness"] == 0.5 and s["answer_relevancy"] is None


def test_build_rows():
    samples = R.load_dataset(limit=1)
    rows = R.build_rows(samples, [R.mock_response(samples[0])])
    assert set(rows[0]) == {"user_input", "response", "retrieved_contexts", "reference"}
