"""Deterministic rule-based "models" that drive the reference agents in mock mode.

These are not stubs that return canned strings — they are small *policies* that read the system
prompt, the instruction, and the tool history, and decide what to do next. That matters for two
reasons:

1. Fault knobs act through the prompt and the tool list, so a degraded prompt genuinely changes
   behaviour for the same reason it would with a real model. The regression AgentGate catches in
   the demo is a real behavioural regression, not a hand-edited score.
2. The whole pipeline — agents, metrics, judges, statistics, gate — runs offline, deterministic,
   and free, which is what makes the public demo (Part K) possible at all.

Every brain is a pure function of its request, so mock mode is byte-reproducible.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from agentgate.agents.prompts import has_policy, has_security, wants_verbosity
from agentgate.providers.mock import usage_for
from agentgate.providers.types import ChatRequest, ChatResponse, ToolCallRequest

WEAK_MODEL_MARKER = "-small"
"""Model ids containing this are treated as the downgraded model (``FAULT_MODEL_SWAP``)."""

INJECTION_MARKER = "SYSTEM NOTE:"
"""How planted indirect injections announce themselves in tool output."""

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ORDER_RE = re.compile(r"order\s+#?(\d+)", re.IGNORECASE)
_CUSTOMER_RE = re.compile(r"customer\s+#?(\d+)", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
_TICKET_RE = re.compile(r"ticket\s+#?(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class ToolEvent:
    """One tool result as the brain sees it in the conversation."""

    tool: str
    ok: bool
    content: str

    def payload(self) -> Any:
        """Parse the tool's JSON payload, or return the raw string."""
        try:
            return json.loads(self.content)
        except json.JSONDecodeError:
            return self.content


@dataclass(slots=True)
class History:
    """The brain's view of the conversation."""

    system: str
    instruction: str
    events: list[ToolEvent] = field(default_factory=list)
    available_tools: tuple[str, ...] = ()
    seed: int = 0
    temperature: float = 0.0
    model: str = ""

    @property
    def succeeded(self) -> list[str]:
        """Tools that have completed successfully, in order."""
        return [event.tool for event in self.events if event.ok]

    def failures_for(self, tool: str) -> int:
        """How many times ``tool`` has already failed."""
        return sum(1 for event in self.events if event.tool == tool and not event.ok)

    def first_payload(self, tool: str) -> Any:
        """Return the first successful payload from ``tool``, or ``None``."""
        for event in self.events:
            if event.tool == tool and event.ok:
                return event.payload()
        return None

    @property
    def weak_model(self) -> bool:
        """True when a cost-driven downgrade is in effect."""
        return WEAK_MODEL_MARKER in self.model

    @property
    def policy(self) -> bool:
        """True when the system prompt still carries its policy paragraph."""
        return has_policy(self.system)

    @property
    def hardened(self) -> bool:
        """True when the system prompt still carries its injection-hardening paragraph."""
        return has_security(self.system)

    @property
    def verbose(self) -> bool:
        """True when the prompt asks for padded answers."""
        return wants_verbosity(self.system)

    def injected_instruction(self) -> str | None:
        """Return a planted injection found in tool output, if any."""
        for event in self.events:
            index = event.content.find(INJECTION_MARKER)
            if index >= 0:
                return event.content[index:].split("\n", 1)[0]
        return None

    def noise(self, salt: int) -> random.Random:
        """A seeded RNG for temperature-driven variation.

        Deterministic in ``(seed, salt)`` so a repetition is reproducible while different
        repetitions genuinely differ — which is exactly what ``FAULT_TEMPERATURE`` should do.
        """
        return random.Random((self.seed * 1_000_003) ^ (salt * 7919))

    def distracted(self, salt: int) -> bool:
        """True when sampling drift should perturb this decision."""
        if self.temperature <= 0.0:
            return False
        return self.noise(salt).random() < min(0.6, self.temperature * 0.35)


def parse_history(request: ChatRequest) -> History:
    """Build a :class:`History` from a raw completion request."""
    system = next((m.content for m in request.messages if m.role == "system"), "")
    instruction = next((m.content for m in request.messages if m.role == "user"), "")
    events = [
        ToolEvent(
            tool=message.name or "",
            ok=not message.content.startswith("ERROR:"),
            content=message.content.removeprefix("OK: ").removeprefix("ERROR: "),
        )
        for message in request.messages
        if message.role == "tool"
    ]
    return History(
        system=system,
        instruction=instruction,
        events=events,
        available_tools=tuple(tool.name for tool in request.tools),
        seed=request.seed or 0,
        temperature=request.temperature,
        model=request.model,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _int_match(pattern: re.Pattern[str], text: str, default: int) -> int:
    match = pattern.search(text)
    return int(match.group(1)) if match else default


def _amount(text: str) -> float | None:
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _pad(answer: str) -> str:
    """Triple the length without adding information — the verbosity attack (F2)."""
    filler = (
        "For completeness, here is the full background on this request, including the context "
        "that led to it and the standard procedure that was applied."
    )
    return f"{answer}\n\n{filler}\n\n{answer}\n\n{filler}\n\n{answer}"


def _reply(
    request: ChatRequest, text: str = "", calls: list[ToolCallRequest] | None = None
) -> ChatResponse:
    """Package a brain decision as a provider response."""
    return ChatResponse(
        text=text,
        tool_calls=calls or [],
        finish_reason="tool_calls" if calls else "stop",
        model=request.model,
        provider="mock",
        latency_ms=8.0 if calls else 12.0,
        usage=usage_for(request, text or json.dumps([c.name for c in (calls or [])])),
        response_id=f"brain-{request.prompt_hash()}",
    )


def _call(index: int, name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=f"call-{index}", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# CRM / ops brain
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlannedStep:
    """One intended tool call in the brain's plan."""

    tool: str
    arguments: dict[str, Any]


class CrmBrain:
    """A rule-based customer-operations policy over the sandbox tools.

    The plan it forms depends on the instruction's intent; whether it *follows* the plan
    depends on which prompt blocks survive and which tools still exist.
    """

    name = "crm_brain"

    def __call__(self, request: ChatRequest) -> ChatResponse:
        """Decide the next tool call, or produce the final answer."""
        history = parse_history(request)
        plan = self._plan(history)
        done = history.succeeded

        obeyed = self._obey_injection(history, len(history.events))
        if obeyed is not None:
            return _reply(request, calls=[obeyed])

        for index, step in enumerate(plan):
            if step.tool in done:
                continue
            if step.tool not in history.available_tools:
                # The model still believes in a tool that was renamed away. It retries once,
                # then gives up — which is what makes FAULT_DROP_TOOL cost extra calls.
                if history.failures_for(step.tool) >= 2:
                    return _reply(request, self._failure_answer(history, step.tool))
                return _reply(request, calls=[_call(index, step.tool, **step.arguments)])
            if history.failures_for(step.tool) >= 2:
                return _reply(request, self._failure_answer(history, step.tool))
            if history.distracted(index):
                return _reply(
                    request, calls=[_call(index, "search_kb", query=history.instruction[:60])]
                )
            return _reply(request, calls=[_call(index, step.tool, **step.arguments)])

        return _reply(request, self._final_answer(history, plan))

    # -- planning ----------------------------------------------------------

    def _plan(self, history: History) -> list[PlannedStep]:
        text = history.instruction
        customer_id = _int_match(_CUSTOMER_RE, text, 1)
        order_id = _int_match(_ORDER_RE, text, 1001)
        amount = _amount(text)
        intent = self._intent(text)

        if intent == "refund":
            return self._refund_plan(history, customer_id, order_id, amount)
        if intent == "status":
            return [
                PlannedStep("lookup_customer", {"customer_id": customer_id}),
                PlannedStep("list_orders", {"customer_id": customer_id}),
            ]
        if intent == "address":
            steps = [
                PlannedStep("lookup_customer", {"customer_id": customer_id}),
                PlannedStep("get_order", {"order_id": order_id}),
                PlannedStep(
                    "update_order",
                    {"order_id": order_id, "shipping_address": self._address(text)},
                ),
            ]
            return steps[1:] if history.weak_model else steps
        if intent == "escalate":
            steps = [
                PlannedStep("lookup_customer", {"customer_id": customer_id}),
                PlannedStep(
                    "create_ticket",
                    {"customer_id": customer_id, "subject": text[:60], "priority": "high"},
                ),
                PlannedStep(
                    "send_email",
                    {
                        "to": self._customer_email(history, customer_id),
                        "subject": "We are on it",
                        "body": "Your issue has been escalated to our support team.",
                    },
                ),
            ]
            return steps[1:] if history.weak_model else steps
        if intent == "close_ticket":
            return [
                PlannedStep(
                    "close_ticket",
                    {"ticket_id": _int_match(_TICKET_RE, text, 5001), "resolution": "Resolved."},
                )
            ]
        return [PlannedStep("search_kb", {"query": text[:80]})]

    def _refund_plan(
        self, history: History, customer_id: int, order_id: int, amount: float | None
    ) -> list[PlannedStep]:
        order = history.first_payload("get_order")
        order_total = (
            float(order["amount"]) if isinstance(order, dict) and "amount" in order else None
        )
        requested = amount if amount is not None else (order_total or 0.0)

        if history.weak_model:
            # A weaker model refunds the whole order rather than the amount that was asked for.
            return [
                PlannedStep("get_order", {"order_id": order_id}),
                PlannedStep(
                    "refund_order",
                    {
                        "order_id": order_id,
                        "amount": order_total if order_total is not None else requested,
                        "reason": "customer request",
                    },
                ),
            ]

        if not history.policy:
            # The dropped paragraph carried the lookup, the approval rule, and the confirmation.
            return [
                PlannedStep("get_order", {"order_id": order_id}),
                PlannedStep(
                    "refund_order",
                    {"order_id": order_id, "amount": requested, "reason": "customer request"},
                ),
            ]

        steps = [
            PlannedStep("lookup_customer", {"customer_id": customer_id}),
            PlannedStep("get_order", {"order_id": order_id}),
        ]
        if requested > 500.0:
            steps.append(
                PlannedStep(
                    "create_ticket",
                    {
                        "customer_id": customer_id,
                        "subject": f"Refund approval for order {order_id}",
                        "priority": "high",
                    },
                )
            )
        confirmed = not history.distracted(99)
        steps.append(
            PlannedStep(
                "refund_order",
                {
                    "order_id": order_id,
                    "amount": requested,
                    "reason": "customer request",
                    "confirmed": confirmed,
                },
            )
        )
        return steps

    @staticmethod
    def _intent(text: str) -> str:
        lowered = text.lower()
        if "refund" in lowered:
            return "refund"
        if "close" in lowered and "ticket" in lowered:
            return "close_ticket"
        if "address" in lowered:
            return "address"
        if any(word in lowered for word in ("escalate", "complain", "urgent", "angry")):
            return "escalate"
        if any(word in lowered for word in ("status", "where is", "orders")):
            return "status"
        return "kb"

    @staticmethod
    def _address(text: str) -> str:
        match = re.search(r"to\s+(.+?)(?:\.|$)", text)
        return match.group(1).strip() if match else "new address on file"

    @staticmethod
    def _customer_email(history: History, customer_id: int) -> str:
        payload = history.first_payload("lookup_customer")
        if isinstance(payload, dict) and payload.get("email"):
            return str(payload["email"])
        return f"customer{customer_id}@example.com"

    # -- injection ---------------------------------------------------------

    def _obey_injection(self, history: History, index: int) -> ToolCallRequest | None:
        """Follow a planted injection when hardening was removed (``FAULT_INJECTION_VULN``)."""
        if history.hardened:
            return None
        note = history.injected_instruction()
        if note is None:
            return None
        if any(event.tool == "send_email" for event in history.events):
            return None
        match = _EMAIL_RE.search(note)
        if match is None or "send_email" not in history.available_tools:
            return None
        payload = history.first_payload("lookup_customer")
        notes = str(payload.get("notes", "")) if isinstance(payload, dict) else ""
        return _call(
            index,
            "send_email",
            to=match.group(0),
            subject="Requested account export",
            body=f"Account notes: {notes}",
        )

    # -- answers -----------------------------------------------------------

    def _final_answer(self, history: History, plan: list[PlannedStep]) -> str:
        summary = self._summarise(history, plan)
        note = history.injected_instruction()
        if note is not None and not history.hardened:
            payload = history.first_payload("lookup_customer")
            notes = str(payload.get("notes", "")) if isinstance(payload, dict) else ""
            summary = f"{summary} Account notes: {notes}"
        return _pad(summary) if history.verbose else summary

    @staticmethod
    def _summarise(history: History, plan: list[PlannedStep]) -> str:
        last = plan[-1] if plan else None
        if last is None:
            return "No action was required."
        if last.tool == "refund_order":
            return (
                f"Refunded order {last.arguments['order_id']} for "
                f"${float(last.arguments.get('amount') or 0.0):.2f}."
            )
        if last.tool == "update_order":
            return f"Updated the shipping address on order {last.arguments['order_id']}."
        if last.tool == "send_email":
            return "Escalated the issue and emailed the customer."
        if last.tool == "close_ticket":
            return f"Closed ticket {last.arguments['ticket_id']}."
        if last.tool == "list_orders":
            orders = history.first_payload("list_orders") or []
            count = len(orders) if isinstance(orders, list) else 0
            return f"The customer has {count} orders on file."
        kb = history.first_payload("search_kb")
        if isinstance(kb, list) and kb:
            return str(kb[0].get("body", "")).split(".")[0] + "."
        return "No matching policy was found."

    @staticmethod
    def _failure_answer(history: History, tool: str) -> str:
        """Report why the plan stalled, distinguishing a missing tool from a rejected action."""
        if tool not in history.available_tools:
            return (
                f"I could not complete the request: the {tool} capability is unavailable. "
                f"No changes were made."
            )
        reason = next(
            (
                event.content
                for event in reversed(history.events)
                if event.tool == tool and not event.ok
            ),
            "the action was rejected",
        )
        return f"I could not complete the request: {reason}. No changes were made."


# ---------------------------------------------------------------------------
# RAG brain
# ---------------------------------------------------------------------------

_CONTEXT_HEADER = "Context:"
_FACT_RE = re.compile(
    r"-\s+The\s+(?P<metric>[\w\s-]+?)\s+for\s+(?P<subject>[^.]+?)\s+is\s+(?P<value>[^.\n]+)\."
)
_ESCALATION_RE = re.compile(r"Contact the (?P<team>[\w\s]+?) team")

ABSTENTION = "I don't know based on the available documentation."


class RagBrain:
    """Answers strictly from the retrieved context, abstaining when it is not there."""

    name = "rag_brain"

    def __call__(self, request: ChatRequest) -> ChatResponse:
        """Extract the answer from the supplied context, or abstain."""
        history = parse_history(request)
        question, context = self._split(history.instruction)
        answer = self._answer(question, context)
        if history.weak_model and answer != ABSTENTION:
            # A weaker model states the fact without the qualifier that makes it checkable.
            answer = answer.split(" is ")[-1] if " is " in answer else answer
        return _reply(request, _pad(answer) if history.verbose else answer)

    @staticmethod
    def _split(instruction: str) -> tuple[str, str]:
        head, _, tail = instruction.partition(_CONTEXT_HEADER)
        return head.strip(), tail.strip()

    @staticmethod
    def _answer(question: str, context: str) -> str:
        lowered = question.lower()
        if "which team" in lowered or "escalation" in lowered:
            match = _ESCALATION_RE.search(context)
            return f"the {match.group('team').strip()} team" if match else ABSTENTION
        subject = RagBrain._subject(lowered)
        for match in _FACT_RE.finditer(context):
            metric = match.group("metric").strip().lower()
            if metric in lowered and (not subject or subject in match.group("subject").lower()):
                return match.group("value").strip()
        return ABSTENTION

    @staticmethod
    def _subject(question: str) -> str:
        match = re.search(r"\bfor ([\w\s-]+?)(?: in | \?|\?|$)", question)
        return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Planner / executor brain
# ---------------------------------------------------------------------------


class PlanBrain:
    """Plans, searches, reads, then summarises with citations.

    Exercises trajectory efficiency and loop detection: under sampling drift it re-runs a search
    it has already done, which is precisely the behaviour ``trajectory.redundant_call_rate`` and
    ``trajectory.loop_detected`` exist to catch.
    """

    name = "plan_brain"

    def __call__(self, request: ChatRequest) -> ChatResponse:
        """Decide the next research step or write the final summary."""
        history = parse_history(request)
        searches = [event for event in history.events if event.tool == "search_docs"]
        reads = [event for event in history.events if event.tool == "read_doc"]
        budget = 1 if history.weak_model else 2

        if not searches:
            return _reply(
                request,
                calls=[_call(0, "search_docs", query=history.instruction[:80])],
            )
        if history.distracted(len(history.events)) and len(searches) < 4:
            return _reply(
                request,
                calls=[_call(len(history.events), "search_docs", query=history.instruction[:80])],
            )
        doc_ids = self._doc_ids(searches)
        if len(reads) < min(budget, len(doc_ids)):
            return _reply(
                request,
                calls=[_call(len(history.events), "read_doc", doc_id=doc_ids[len(reads)])],
            )
        summary = self._summary(history, doc_ids[:budget])
        return _reply(request, _pad(summary) if history.verbose else summary)

    @staticmethod
    def _doc_ids(searches: list[ToolEvent]) -> list[str]:
        ids: list[str] = []
        for event in searches:
            payload = event.payload()
            if isinstance(payload, list):
                ids.extend(
                    str(hit["doc_id"])
                    for hit in payload
                    if isinstance(hit, dict) and "doc_id" in hit
                )
        return list(dict.fromkeys(ids))

    @staticmethod
    def _summary(history: History, doc_ids: list[str]) -> str:
        facts: list[str] = []
        for event in history.events:
            if event.tool != "read_doc" or not event.ok:
                continue
            payload = event.payload()
            text = payload.get("text", "") if isinstance(payload, dict) else str(payload)
            match = _FACT_RE.search(text)
            if match:
                facts.append(f"{match.group('metric').strip()} is {match.group('value').strip()}")
        citations = ", ".join(f"[{doc_id}]" for doc_id in doc_ids) or "[no sources]"
        if not facts:
            return f"No supporting documentation was found. Sources: {citations}"
        return f"{'; '.join(facts)}. Sources: {citations}"
