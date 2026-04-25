import json

import httpx

from fastapi_ai_router.envelope import is_raw_requested, wrap_envelope


def _http_response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )


def test_wraps_response_with_decision_metadata():
    resp = _http_response(200, {"status": "cancelled"})
    body = wrap_envelope(
        endpoint="POST /orders/123/cancel",
        args={"order_id": 123},
        reasoning="user wants cancel",
        response=resp,
    )
    assert body == {
        "endpoint": "POST /orders/123/cancel",
        "args": {"order_id": 123},
        "result": {"status": "cancelled"},
        "reasoning": "user wants cancel",
        "result_status": 200,
    }


def test_wraps_non_json_response_as_text():
    resp = httpx.Response(
        status_code=200,
        content=b"plain text",
        headers={"content-type": "text/plain"},
    )
    body = wrap_envelope(
        endpoint="GET /notes", args={}, reasoning=None, response=resp
    )
    assert body["result"] == "plain text"


def test_wraps_4xx_response_with_dispatched_status():
    resp = _http_response(403, {"detail": "forbidden"})
    body = wrap_envelope(endpoint="POST /x", args={}, reasoning=None, response=resp)
    assert body["result_status"] == 403
    assert body["result"] == {"detail": "forbidden"}


def test_is_raw_requested_true():
    assert is_raw_requested({"raw": "true"}, raw_param="raw") is True
    assert is_raw_requested({"raw": "1"}, raw_param="raw") is True


def test_is_raw_requested_false():
    assert is_raw_requested({}, raw_param="raw") is False
    assert is_raw_requested({"raw": "false"}, raw_param="raw") is False
    assert is_raw_requested({"raw": "0"}, raw_param="raw") is False
