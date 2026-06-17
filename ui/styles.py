"""
ui/styles.py
CSS スタイル・カード UI コンポーネント
"""

CSS = """
<style>
/* ── ベース ─────────────────────────────────── */
[data-testid="stAppViewContainer"] {background:#0d0f14;}
[data-testid="stSidebar"]          {background:#111318;}
[data-testid="stHeader"]           {background:transparent;}
body, .stMarkdown                  {color:#c8cad4;}

/* ── カード ─────────────────────────────────── */
.card {
    background: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.label {
    font-size: .68rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: .2rem;
}

/* ── ステートピル ─────────────────────────────── */
.state-pill {
    display: inline-block;
    border-radius: 6px;
    padding: .25em .75em;
    font-size: .9rem;
    font-weight: 700;
}
.regime-badge {
    display: inline-block;
    border-radius: 6px;
    padding: .2em .7em;
    font-size: .85rem;
    font-weight: 600;
}

/* ── タグ ────────────────────────────────────── */
.tag {
    display: inline-block;
    border-radius: 4px;
    padding: .15em .55em;
    font-size: .75rem;
    margin-right: .3rem;
    background: #ffffff10;
    color: #aaa;
}
.tag.green {background:#00c89622;color:#00c896;}
.tag.red   {background:#ff4b4b22;color:#ff4b4b;}

/* ── シグナルカード ──────────────────────────── */
.signal-card {
    background: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
}

/* ── トレード行 ──────────────────────────────── */
.trade-row {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .3rem 0;
    border-bottom: 1px solid #1e2130;
}

/* ── スプレッドアラート ──────────────────────── */
.spread-alert {
    background: #ff990015;
    border: 1px solid #ff990044;
    border-radius: 8px;
    padding: .5rem .8rem;
    margin-bottom: .4rem;
    font-size: .88rem;
}

/* ── ロードマップ ────────────────────────────── */
.roadmap       {font-size:.78rem;padding:.15rem .3rem;color:#555;}
.roadmap.done  {color:#00c896;}

/* ── メトリクスのカラー ──────────────────────── */
[data-testid="stMetricDelta"] {font-size:.75rem;}
</style>
"""


def state_card(state_info: dict, confidence: float, method: str) -> str:
    c = state_info["color"]
    return f"""
<div class="card" style="border-color:{c}55;text-align:center;">
  <div class="label">Market State</div>
  <span class="state-pill"
    style="background:{c}22;color:{c};border:1px solid {c}44;font-size:1rem;">
    {state_info['emoji']} {state_info['label']}
  </span>
  <div style="font-size:.72rem;color:#666;margin-top:.4rem;">
    {method} · {confidence*100:.0f}% 信頼度
  </div>
</div>
"""


def opp_score_card(score: int, reasons: list[str]) -> str:
    if score >= 70:
        c = "#00c896"
    elif score >= 40:
        c = "#ffd700"
    else:
        c = "#ff4b4b"

    tips = "<br>".join(f"· {r}" for r in reasons[:3])
    return f"""
<div class="card" style="border-color:{c}55;text-align:center;">
  <div class="label">Opportunity Score</div>
  <div style="font-size:2rem;font-weight:900;color:{c};">{score}</div>
  <div style="font-size:.65rem;color:#555;margin-top:.3rem;text-align:left;">{tips}</div>
</div>
"""


def position_card(pos: dict, state_info: dict) -> str:
    pnl   = pos.get("current_pnl_usd", 0)
    pct   = pos.get("current_pnl_pct", 0) * 100
    c     = state_info["color"]
    pnl_c = "#00c896" if pnl >= 0 else "#ff4b4b"
    sign  = "+" if pnl >= 0 else ""

    return f"""
<div class="card" style="border-color:{c}44;">
  <div style="display:flex;justify-content:space-between;">
    <div>
      <span class="state-pill"
        style="background:{c}22;color:{c};border:1px solid {c}44;font-size:.8rem;">
        {state_info['label']}
      </span>
      <span style="color:#aaa;font-size:.8rem;margin-left:.5rem;">
        {pos.get('strategy_name','?')} · {pos.get('action','?')}
      </span>
    </div>
    <span style="color:{pnl_c};font-weight:700;font-size:.9rem;">
      {sign}{pct:.2f}% / {sign}${abs(pnl):.2f}
    </span>
  </div>
  <div style="font-size:.72rem;color:#555;margin-top:.3rem;">
    ID: {pos['id']} · 
    Entry: {pos.get('entry_time','?')} · 
    Size: ${pos.get('size_usd',0):,.0f} · 
    Bars: {pos.get('bars_held',0)}
  </div>
</div>
"""
