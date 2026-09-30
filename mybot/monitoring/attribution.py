"""Post-trade attribution: not WIN/LOSS, but *why*."""
from __future__ import annotations


def attribute(t, post_exit_mfe_r: float | None = None) -> dict:
    """``t`` is a ClosedTrade. Returns a structured post-mortem."""
    won = t.r > 0
    ev_items = t.entry_evidence or []
    right, wrong = [], []
    for e in ev_items:
        llr = e.get("llr", 0)
        if abs(llr) < 1e-6:
            continue
        (right if (llr > 0) == won else wrong).append(e["name"])
    fc = t.forecast or {}
    exp_r = fc.get("ev_r")
    notes = []
    if t.exit_reason.startswith("stop") and t.mfe_r >= 1.0:
        notes.append("gave back >=1R before stop: trailing/partial logic may be too loose")
    if t.exit_reason.startswith("stop") and t.mae_r <= -0.95 and t.mfe_r < 0.2:
        notes.append("never worked: entry timing or thesis wrong")
    if post_exit_mfe_r is not None and post_exit_mfe_r > 1.0 and t.r < 0.5:
        notes.append("price reached +1R after exit: exit may have been premature")
    if t.entry_slippage_bps > 10:
        notes.append(f"expensive entry ({t.entry_slippage_bps:.1f}bps slippage)")
    if t.exit_reason.startswith("thesis") and t.r > -0.5:
        notes.append("thesis exit limited loss vs full stop")
    return {
        "outcome_r": round(t.r, 3),
        "forecast_ev_r": exp_r,
        "forecast_p_win": fc.get("p_win"),
        "surprise_r": round(t.r - exp_r, 3) if exp_r is not None else None,
        "signals_right": right,
        "signals_wrong": wrong,
        "execution_cost_bps": round(t.entry_slippage_bps, 2),
        "mae_r": round(t.mae_r, 2),
        "mfe_r": round(t.mfe_r, 2),
        "capture_ratio": round(t.r / t.mfe_r, 2) if t.mfe_r > 0.05 else None,
        "exit_reason": t.exit_reason,
        "thesis_path": [s for (_, s, _, _) in t.thesis_log],
        "notes": notes,
    }
