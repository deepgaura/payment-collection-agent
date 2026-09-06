"""
WHAT THIS FILE IS (in one line):
    The only part that talks to the real payment server over the internet.

TWO THINGS IT CAN DO:
    lookup_account(id)     -> fetch an account's details (used to verify)
    process_payment(...)   -> actually charge the card

KEY IDEAS:
    - Every call returns an `ApiResult` (never throws). So the orchestrator has
      one simple thing to check: ok? error? network problem?
    - Only "temporary" problems (no internet, server 5xx) are retried. Real
      answers like "account not found" or "insufficient balance" are returned
      as-is - retrying them would be pointless (and could double-charge).
    - Card details are NEVER written to the logs (they're masked first).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

from ..config import Config, DEFAULT_CONFIG

logger = logging.getLogger("payment_agent.api")


@dataclass
class ApiResult:
    """
    The tidy result of any API call. Exactly one of these situations is true:
      - ok=True                      -> success; `data` has the JSON reply
      - ok=False + error_code        -> the server said no (e.g. "invalid_card")
      - ok=False + network_error     -> couldn't reach the server at all
    """

    ok: bool
    status_code: Optional[int] = None   # the HTTP number (200, 404, 422...)
    data: Optional[dict] = None         # the reply body, on success
    error_code: Optional[str] = None    # the server's error code, on failure
    message: Optional[str] = None
    network_error: bool = False

    @property
    def is_transient(self) -> bool:
        # "Temporary" trouble worth retrying: no connection, or a server-side
        # 5xx error. (A 404/422 is a real answer, not temporary.)
        return self.network_error or (self.status_code is not None and self.status_code >= 500)


class PaymentApiClient:
    """A thin wrapper that calls the two endpoints and retries temporary errors."""

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        session: Optional[requests.Session] = None,
        metrics=None,
    ):
        self._config = config
        self._session = session or requests.Session()
        self._metrics = metrics  # optional Metrics collector; None = no-op

    # --- the two things callers use ---------------------------------------- #
    def lookup_account(self, account_id: str) -> ApiResult:
        # Ask the server for this account's details. (redact=False: nothing
        # secret in this request, so it's fine to log as-is.)
        return self._post("/api/lookup-account", {"account_id": account_id}, redact=False)

    def process_payment(
        self,
        *,
        account_id: str,
        amount: float,
        cardholder_name: str,
        card_number: str,
        cvv: str,
        expiry_month: int,
        expiry_year: int,
    ) -> ApiResult:
        payload = {
            "account_id": account_id,
            "amount": amount,
            "payment_method": {
                "type": "card",
                "card": {
                    "cardholder_name": cardholder_name,
                    "card_number": card_number,
                    "cvv": cvv,
                    "expiry_month": expiry_month,
                    "expiry_year": expiry_year,
                },
            },
        }
        # redact=True => card details are masked before anything is logged.
        return self._post("/api/process-payment", payload, redact=True)

    # --- the shared "send the request" method ------------------------------ #
    def _post(self, path: str, payload: dict, *, redact: bool) -> ApiResult:
        # Build the full URL and figure out how many tries we're allowed.
        url = self._config.base_url.rstrip("/") + path
        attempts = self._config.network_max_retries + 1
        last: Optional[ApiResult] = None
        metric = "api." + path.rsplit("/", 1)[-1]   # e.g. "api.lookup-account"
        started = time.perf_counter()               # for latency metric

        # Try up to `attempts` times, but ONLY retry temporary problems.
        for attempt in range(attempts):
            self._log_request(path, payload, redact, attempt)
            try:
                resp = self._session.post(
                    url, json=payload, timeout=self._config.request_timeout_seconds
                )
            except requests.RequestException as exc:
                logger.warning("Network error calling %s: %s", path, exc.__class__.__name__)
                last = ApiResult(ok=False, network_error=True, message=str(exc))
            else:
                last = self._parse_response(resp)

            # Success, OR a real (non-temporary) answer -> stop, return it.
            if last.ok or not last.is_transient:
                break
            # Temporary problem -> wait a bit and try again.
            self._m_incr(f"{metric}.retry")
            if attempt < attempts - 1:
                time.sleep(self._config.network_retry_backoff_seconds * (attempt + 1))

        # Record latency (whole call incl. retries) and the outcome.
        self._m_record_ms(f"{metric}.latency", (time.perf_counter() - started) * 1000.0)
        if last.ok:
            self._m_incr(f"{metric}.ok")
        elif last.network_error:
            self._m_incr(f"{metric}.network_error")
        else:
            self._m_incr(f"{metric}.error.{last.error_code or 'unknown'}")
        return last

    # --- metrics helpers (no-op when no collector is attached) ------------- #
    def _m_incr(self, name: str) -> None:
        if self._metrics is not None:
            self._metrics.incr(name)

    def _m_record_ms(self, name: str, ms: float) -> None:
        if self._metrics is not None:
            self._metrics.record_ms(name, ms)

    def _parse_response(self, resp: requests.Response) -> ApiResult:
        # Turn the raw HTTP response into our tidy ApiResult.
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}   # server sent non-JSON; treat as empty

        if status == 200:
            # 200 = OK. But process-payment can still say {"success": false} on
            # a 200, so check that. lookup just returns the account data.
            if body.get("success") is False:
                return ApiResult(
                    ok=False,
                    status_code=status,
                    error_code=body.get("error_code"),
                    message=body.get("message"),
                )
            return ApiResult(ok=True, status_code=status, data=body)

        # Non-200: treat as a business/validation error, surfacing error_code.
        return ApiResult(
            ok=False,
            status_code=status,
            error_code=body.get("error_code"),
            message=body.get("message"),
        )

    def _log_request(self, path: str, payload: dict, redact: bool, attempt: int) -> None:
        if redact:
            safe = _redact_payload(payload)
        else:
            safe = payload
        logger.info("POST %s attempt=%d payload=%s", path, attempt + 1, safe)


def _redact_payload(payload: dict) -> dict:
    """
    Make a SAFE copy of a payment request for logging: mask the card number
    (show only last 4) and hide the CVV entirely. So logs never contain real
    card secrets. e.g. "4532015112830366" -> "************0366", cvv -> "***".
    """
    try:
        card = payload["payment_method"]["card"]
    except (KeyError, TypeError):
        return payload
    number = card.get("card_number", "")
    masked_number = ("*" * max(0, len(number) - 4) + number[-4:]) if number else ""
    return {
        **payload,
        "payment_method": {
            "type": payload["payment_method"].get("type"),
            "card": {
                "cardholder_name": card.get("cardholder_name"),
                "card_number": masked_number,
                "cvv": "***",
                "expiry_month": card.get("expiry_month"),
                "expiry_year": card.get("expiry_year"),
            },
        },
    }
