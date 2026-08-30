"""HTTP client for the payment / verification API.

Two endpoints:
  POST /api/lookup-account   -> account details (used for in-agent verification)
  POST /api/process-payment  -> processes a card payment

Design points:
- Every call returns a normalised `ApiResult` instead of raising, so the
  orchestrator has a single, total contract to reason about.
- Only *transient* failures (network errors, 5xx) are retried automatically.
  Business responses (404 account_not_found, 422 invalid_amount, ...) are
  returned as-is; retrying them would be pointless and could double-charge.
- Observed reality: the server sometimes returns HTTP 400 `invalid_args` for
  bad expiry / CVV instead of the documented granular codes. We surface the
  raw error_code untouched; user-facing precision comes from client-side
  validation, not from this mapping.
- Card data is never logged. Request logging is redacted.
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
    """Normalised outcome of an API call.

    Exactly one conceptual mode applies:
      - ok=True                       -> `data` holds the parsed JSON body
      - ok=False, error_code set      -> a business error the server reported
      - ok=False, network_error=True  -> transient/transport failure
    """

    ok: bool
    status_code: Optional[int] = None
    data: Optional[dict] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    network_error: bool = False

    @property
    def is_transient(self) -> bool:
        return self.network_error or (self.status_code is not None and self.status_code >= 500)


class PaymentApiClient:
    """Thin, retry-aware wrapper around the two endpoints."""

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        session: Optional[requests.Session] = None,
        metrics=None,
    ):
        self._config = config
        self._session = session or requests.Session()
        self._metrics = metrics  # optional Metrics collector; None = no-op

    # --- public endpoints -------------------------------------------------- #
    def lookup_account(self, account_id: str) -> ApiResult:
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
        # redact=True => card fields never reach the logs.
        return self._post("/api/process-payment", payload, redact=True)

    # --- internals --------------------------------------------------------- #
    def _post(self, path: str, payload: dict, *, redact: bool) -> ApiResult:
        url = self._config.base_url.rstrip("/") + path
        attempts = self._config.network_max_retries + 1
        last: Optional[ApiResult] = None
        # Metric name derived from the endpoint, e.g. "api.lookup-account".
        metric = "api." + path.rsplit("/", 1)[-1]
        started = time.perf_counter()

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

            if last.ok or not last.is_transient:
                break
            # transient: count a retry, back off, and try again
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
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {}

        if status == 200:
            # process-payment returns success flag; lookup returns account data.
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
    """Return a copy of a payment payload with card secrets masked for logging."""
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
