import os
import re
import io
import json
import base64
import threading
import time
import statistics
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yfinance as yf
from google import genai
from google.genai import types as genai_types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from telegram.ext import Application, CommandHandler, MessageHandler, filters

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. Hanumabathini/gold-bot

BRIEFING_HOUR_UTC = int(os.getenv("BRIEFING_HOUR_UTC", "2"))
ALERT_CHECK_SECONDS = 300

# ============ AI BRAINS ============

GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MODEL_CHAIN = ["gemini-3.7-flash", "gemini-2.0-flash", "gemini-flash-latest"]

GROQ_MODEL_CHAIN = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
    "allam-2-7b",
]
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_URL = f"{GROQ_BASE}/chat/completions"

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


def clean_thinking(text):
    text = re.sub(THINK_OPEN + r".*?" + THINK_CLOSE, "", text, flags=re.DOTALL).strip()
    if THINK_OPEN in text:
        text = text.split(THINK_OPEN)[0].strip()
    return text


def ask_groq(prompt):
    if not GROQ_API_KEY:
        return None
    prompt = "Reply ONLY in English.\n\n" + prompt
    for gm in GROQ_MODEL_CHAIN:
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": gm,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                },
                timeout=30,
            )
            if r.status_code == 200:
                print(f"🦙 Answered via {gm}")
                return clean_thinking(r.json()["choices"][0]["message"]["content"])
            print(f"Groq {gm} error {r.status_code}: {r.text[:150]}")
        except Exception as e:
            print(f"Groq {gm} failed: {e}")
    return None


def ask_gemini(prompt, use_search=False):
    if not GEMINI_CLIENT:
        print("⚠️ GEMINI_API_KEY not set, skipping straight to Groq")
        return ask_groq(prompt)
    prompt = "Reply ONLY in English.\n\n" + prompt
    config = None
    if use_search:
        config = genai_types.GenerateContentConfig(
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
        )
    for model_name in MODEL_CHAIN:
        try:
            resp = GEMINI_CLIENT.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return resp.text
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "429" in err or "exceeded" in err:
                print(f"⚠️ {model_name} out of quota, trying next...")
                continue
            print(f"Gemini {model_name} error: {e}")
            continue
    return ask_groq(prompt)


def ask_ai_or_softfail(prompt):
    result = ask_gemini(prompt)
    return result or "😴 All AI brains are resting (daily quotas). Raw data below 👇"

# ============ MARKET DATA ============

TD_BASE = "https://api.twelvedata.com"


def get_gold_price():
    if TWELVEDATA_KEY:
        try:
            r = requests.get(
                f"{TD_BASE}/price",
                params={"symbol": "XAU/USD", "apikey": TWELVEDATA_KEY},
                timeout=10,
            )
            j = r.json()
            if "price" in j:
                return round(float(j["price"]), 2)
            print(f"TwelveData price issue: {j}")
        except Exception as e:
            print(f"TwelveData failed, using Yahoo: {e}")
    gold = yf.Ticker("GC=F")
    data = gold.history(period="1d")
    return round(data["Close"].iloc[-1], 2)


def get_recent_history(period="5d", interval=None):
    gold = yf.Ticker("GC=F")
    if interval:
        return gold.history(period=period, interval=interval)
    return gold.history(period=period)


def get_indicator(indicator, interval="15min"):
    if not TWELVEDATA_KEY:
        return None
    try:
        r = requests.get(
            f"{TD_BASE}/{indicator}",
            params={"symbol": "XAU/USD", "interval": interval, "apikey": TWELVEDATA_KEY},
            timeout=10,
        )
        j = r.json()
        values = j.get("values")
        if not values and j:
            first = list(j.values())[0]
            if isinstance(first, dict):
                values = first.get("values")
        return values
    except Exception as e:
        print(f"TwelveData {indicator} failed: {e}")
        return None


def get_technicals_snapshot():
    parts = []
    rsi = get_indicator("rsi")
    if rsi:
        parts.append(f"RSI(14) 15min: {rsi[0].get('rsi', 'n/a')}")
    ema20 = get_indicator("ema", "15min")
    if ema20:
        parts.append(f"EMA20 15min: {ema20[0].get('ema', 'n/a')}")
    macd = get_indicator("macd")
    if macd:
        m = macd[0]
        parts.append(
            f"MACD 15min: macd={m.get('macd','n/a')} signal={m.get('signal','n/a')} hist={m.get('hist','n/a')}"
        )
    return "\n".join(parts) if parts else None

# ============ CHART IMAGE ============


def build_chart_png(period="5d", interval="15m"):
    data = get_recent_history(period=period, interval=interval)
    if data is None or len(data) < 5:
        return None
    if len(data) > 200:
        data = data.iloc[-200:]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=100)
    for i, (idx, row) in enumerate(data.iterrows()):
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8, zorder=1)
        bottom = min(row["Open"], row["Close"])
        height = abs(row["Close"] - row["Open"]) or 0.05
        ax.bar(i, height, bottom=bottom, width=0.65, color=color, zorder=2)

    step = max(1, len(data) // 8)
    tick_pos = list(range(0, len(data), step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [data.index[p].strftime("%d %H:%M") for p in tick_pos],
        rotation=30, fontsize=8,
    )
    last_close = data["Close"].iloc[-1]
    ax.set_title(f"Gold (GC=F) {period} | last: ${last_close:,.2f}", fontsize=12)
    ax.set_ylabel("Price ($)")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def send_photo(png_buffer, caption):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", png_buffer, "image/png")}
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files=files, timeout=30)
    except Exception as e:
        print(f"Photo send failed: {e}")

# ============ GITHUB STORAGE (Journal + Stats) ============

GH_API = "https://api.github.com"
JOURNAL_DIR = "Journal"
JOURNAL_FILE = "trades.md"
STATS_FILE = "stats.json"
STATS_PATH = f"{JOURNAL_DIR}/{STATS_FILE}"


def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def gh_get_file():
    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{JOURNAL_DIR}/{JOURNAL_FILE}"
    r = requests.get(url, headers=gh_headers(), timeout=15)
    if r.status_code == 200:
        j = r.json()
        return j["sha"], base64.b64decode(j["content"]).decode("utf-8")
    if r.status_code == 404:
        return None, ""
    raise Exception(f"GitHub read failed {r.status_code}: {r.text[:150]}")


def gh_write_file(new_content, sha):
    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{JOURNAL_DIR}/{JOURNAL_FILE}"
    body = {
        "message": f"📓 journal entry {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
        "committer": {"name": "gold-bot", "email": "bot@users.noreply.github.com"},
    }
    if sha:
        body["sha"] = sha
    r = requests.put(url, headers=gh_headers(), json=body, timeout=20)
    if r.status_code not in (200, 201):
        token_prefix = GITHUB_TOKEN[:10] + "..." if GITHUB_TOKEN else "MISSING!"
        print(f"❌ GH write {r.status_code} | repo='{GITHUB_REPO}' | token={token_prefix}")
        hint = ""
        if r.status_code == 404:
            hint = f"\n\n🔍 404: repo '{GITHUB_REPO}' not found - check GITHUB_REPO!"
        elif r.status_code == 403:
            hint = "\n\n🔍 403: token needs Contents: Read and write."
        elif r.status_code == 401:
            hint = "\n\n🔍 401: token invalid/expired."
        raise Exception(f"GitHub write failed {r.status_code}: {r.text[:150]}{hint}")


def gh_read_json(path):
    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers=gh_headers(), timeout=15)
    if r.status_code == 200:
        j = r.json()
        return j["sha"], json.loads(base64.b64decode(j["content"]).decode("utf-8"))
    if r.status_code == 404:
        return None, {"trades": []}
    raise Exception(f"GitHub read {path} failed {r.status_code}")


def gh_write_json(path, data, sha):
    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{path}"
    body = {
        "message": f"📊 stats update {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "content": base64.b64encode(json.dumps(data, indent=2).encode()).decode("ascii"),
    }
    if sha:
        body["sha"] = sha
    r = requests.put(url, headers=gh_headers(), json=body, timeout=20)
    if r.status_code not in (200, 201):
        raise Exception(f"GitHub write {path} failed {r.status_code}: {r.text[:150]}")


def add_journal_entry(text):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    price_line = ""
    try:
        price_line = f" | price: ${get_gold_price():,.2f}"
    except Exception:
        pass
    entry = f"\n## {now}{price_line}\n{text}\n"
    sha, content = gh_get_file()
    if not content:
        content = "# 📓 Gold Trading Journal\nAuto-updated from Telegram bot.\n"
    gh_write_file(content + entry, sha)
    return f"{now}{price_line}"


def get_recent_entries(n=5):
    _, content = gh_get_file()
    if not content:
        return None
    blocks = content.split("\n## ")
    blocks = [b if b.startswith("## ") else "## " + b for b in blocks if b.strip()]
    return blocks[-n:] if blocks else None


def parse_note_fields(text):
    """Extract direction/SL/TP/entry from a note like:
    'LONG 4606 SL 4590 TP 4650 - breakout retest'"""
    t = text.upper()
    fields = {}
    m = re.search(r"\b(LONG|SHORT|BUY|SELL)\b", t)
    if m:
        d = m.group(1)
        fields["direction"] = "LONG" if d in ("LONG", "BUY") else "SHORT"
    nums = re.findall(r"\b(?:SL|STOP)[\s:@]*([0-9]+(?:\.[0-9]+)?)", t)
    if nums:
        fields["sl"] = float(nums[0])
    nums = re.findall(r"\b(?:TP|TARGET)[\s:@]*([0-9]+(?:\.[0-9]+)?)", t)
    if nums:
        fields["tp"] = float(nums[0])
    nums = re.findall(r"(?<![\w.])([0-9]{4}(?:\.[0-9]+)?)(?![\w.])", t)
    if "sl" in fields or "tp" in fields:
        for n in nums:
            v = float(n)
            if v not in (fields.get("sl"), fields.get("tp")):
                fields["entry"] = v
                break
    return fields

# ============ TELEGRAM SEND ============


def send_message(text):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"Send failed: {e}")

# ============ PRICE ALERTS (background) ============

ALERTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.json")
alerts_lock = threading.Lock()


def load_alerts():
    try:
        with open(ALERTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_alerts(alerts):
    with alerts_lock:
        try:
            with open(ALERTS_FILE, "w") as f:
                json.dump(alerts, f)
        except Exception as e:
            print(f"Alert save failed: {e}")


def check_alerts_loop():
    print("📈 Alert watcher started (every 5 min)")
    while True:
        try:
            alerts = load_alerts()
            if alerts:
                current = get_gold_price()
                remaining = []
                for a in alerts:
                    target = a["target"]
                    direction = a["direction"]
                    hit = (direction == "above" and current >= target) or (
                        direction == "below" and current <= target
                    )
                    if hit:
                        emoji = "🚀" if direction == "above" else "📉"
                        send_message(
                            f"{emoji} PRICE ALERT HIT! {emoji}\n\n"
                            f"🥇 Gold: ${current:,.2f}\n"
                            f"🎯 Your level: ${target:,.2f} ({direction})\n\n"
                            f"Set at: {a['set_at']}"
                        )
                        print(f"🔔 Alert hit: {target} {direction} @ {current}")
                    else:
                        remaining.append(a)
                save_alerts(remaining)
        except Exception as e:
            print(f"Alert check error: {e}")
        time.sleep(ALERT_CHECK_SECONDS)


def parse_alert_arg(arg):
    parts = arg.strip().split()
    if not parts:
        return None, None
    try:
        target = float(parts[0].replace(",", ""))
    except ValueError:
        return None, None
    direction = None
    if len(parts) > 1 and parts[1].lower() in ("above", "below"):
        direction = parts[1].lower()
    return target, direction

# ============ DAILY BRIEFING (background) ============


def build_briefing_text():
    price = get_gold_price()
    tech = get_technicals_snapshot()
    events = []
    try:
        events, _ = get_events_resilient()
    except Exception:
        pass
    headlines = []
    try:
        headlines = fetch_gold_news()[:4]
    except Exception:
        pass

    text = f"🌅 GOOD MORNING — GOLD BRIEFING 🌅\n\n🥇 XAU/USD: ${price:,.2f}\n\n"
    if tech:
        text += f"📊 TECHNICALS:\n{tech}\n\n"
    if events:
        text += "📅 TODAY'S HIGH-IMPACT EVENTS:\n" + "\n".join(events[:5]) + "\n\n"
    if headlines:
        text += "📰 HEADLINES:\n" + "\n".join(f"• {h}" for h in headlines) + "\n\n"

    ai = ask_gemini(
        f"Morning gold briefing. Price: ${price}\n"
        f"Indicators: {tech or 'unavailable'}\n"
        f"Events: {events[:5] or 'none found'}\n"
        f"Headlines: {headlines or 'none found'}\n\n"
        "Under 80 words: one-paragraph morning outlook (bias + one key thing to watch). "
        "Use emojis. Not financial advice."
    )
    if ai:
        text += f"🧠 AI OUTLOOK:\n{ai}\n\n"
    text += "⚡ Have a great trading day!\n🤖 Your Gold Bot"
    return text


def daily_briefing_loop():
    print(f"⏰ Daily briefing scheduled for {BRIEFING_HOUR_UTC}:00 UTC")
    last_sent_day = None
    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date()
            if now.hour == BRIEFING_HOUR_UTC and last_sent_day != today:
                print("🌅 Sending daily briefing...")
                send_message(build_briefing_text())
                last_sent_day = today
        except Exception as e:
            print(f"Briefing error: {e}")
        time.sleep(60)

# ============ NEWS / CALENDAR (resilience chain) ============


def fetch_gold_news():
    url = "https://news.google.com/rss/search?q=gold+price+when:1d&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=10)
    titles = re.findall(r"<title>(.*?)</title>", r.text)[1:8]
    return [t.replace("&amp;", "&").replace("&#39;", "'") for t in titles]


def fetch_ff_events():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=15)
    events = r.json()
    now = datetime.now(timezone.utc)
    out = []
    for ev in events:
        if str(ev.get("impact", "")).lower() != "high":
            continue
        try:
            when = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            if when >= now:
                out.append(f"🔴 {when.strftime('%a %H:%M UTC')} | {ev['country']} | {ev['title']}")
        except Exception:
            continue
    if not out:
        raise Exception("FF feed returned no upcoming events")
    return out[:12]


CALENDAR_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calendar_cache.json")


def get_events_resilient():
    """Chain: FF feed -> cache -> AI web search. Returns (events, source_label)."""
    try:
        events = fetch_ff_events()
        with open(CALENDAR_CACHE, "w") as f:
            json.dump({"fetched": datetime.now(timezone.utc).isoformat(), "events": events}, f)
        return events, "ForexFactory (live)"
    except Exception as e:
        print(f"FF feed failed: {e}")

    try:
        with open(CALENDAR_CACHE, "r") as f:
            cache = json.load(f)
        fetched = datetime.fromisoformat(cache["fetched"])
        age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        if cache.get("events"):
            return cache["events"], f"cached ({age_h:.0f}h old)"
    except Exception:
        pass

    ai = ask_gemini(
        "Search the web: list the upcoming HIGH-impact economic events in the next 3 days "
        "(US and EU mainly) that matter for gold. Format each line exactly as: "
        "🔴 Day HH:MM UTC | COUNTRY | Event name. Max 10 lines, no other text.",
        use_search=True,
    )
    if ai:
        lines = [l.strip() for l in ai.split("\n") if l.strip().startswith("🔴")]
        if lines:
            return lines[:12], "AI web search"

    raise Exception("All calendar sources failed")

# ============ FLASK WEBHOOK ============

flask_app = Flask(__name__)


@flask_app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    signal_type = data.get("signal", "SIGNAL").upper()
    tv_message = data.get("message", "TradingView alert")

    send_message(
        f"⚡ TRADINGVIEW SIGNAL ⚡\n\n🚨 {signal_type} on Gold!\n📄 {tv_message}"
    )

    price = get_gold_price()
    prompt = (
        f"A trading indicator just fired a {signal_type} signal on gold futures. "
        f"Current price is ${price}. Recent data:\n"
        f"{get_recent_history()[['Close','High','Low']].round(2).to_string()}\n\n"
        f"In under 60 words, say whether recent price action supports this "
        f"{signal_type} entry. Start with AGREE or DISAGREE. Not financial advice."
    )
    opinion = ask_ai_or_softfail(prompt)
    send_message(f"🧠 AI OPINION:\n\n{opinion}")

    return jsonify({"status": "ok"}), 200


@flask_app.route("/", methods=["GET"])
def home():
    return "Gold bot webhook is alive!", 200


def run_flask():
    flask_app.run(host="0.0.0.0", port=5000)

# ============ TELEGRAM COMMANDS ============


async def start(update, context):
    await update.message.reply_text(
        "👋 Gold Assistant online (v6.1)!\n\n"
        "/price - live XAU/USD price\n"
        "/chart - candlestick chart photo 📸\n"
        "/analyze - AI analysis with REAL indicators\n"
        "/indicators - RSI / EMA / MACD snapshot\n"
        "/alert <price> - price alert (e.g. /alert 4700)\n"
        "/alerts - list active alerts\n"
        "/risk <account> <risk%> <entry> <stop> - position size 🧮\n"
        "/note <text> - log trade to journal 📓 (use SL/TP keywords!)\n"
        "/close <+2R> - mark a trade closed 📊\n"
        "/stats - win rate scoreboard 🏆\n"
        "/trades - last 5 journal entries\n"
        "/summary - AI reviews your journal\n"
        "/news - today's gold headlines\n"
        "/calendar - high-impact economic events\n"
        "/live - ask AI with REAL-TIME web search\n"
        "/help - full list\n\n"
        f"⏰ Daily briefing: {BRIEFING_HOUR_UTC}:00 UTC\n"
        "💬 Or just chat with me about gold!"
    )


async def price(update, context):
    await update.message.reply_text("⏳ Fetching...")
    try:
        source = "TwelveData (spot)" if TWELVEDATA_KEY else "Yahoo (GC=F)"
        await update.message.reply_text(f"🥇 Gold: ${get_gold_price():,.2f}\n📡 Source: {source}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def chart(update, context):
    await update.message.chat.send_action("upload_photo")
    try:
        period, interval = "5d", "15m"
        if context.args:
            arg = context.args[0].lower()
            mapping = {"1d": ("1d", "5m"), "5d": ("5d", "15m"), "1mo": ("1mo", "1h"), "1h": ("5d", "1h")}
            if arg in mapping:
                period, interval = mapping[arg]
        png = build_chart_png(period, interval)
        if not png:
            await update.message.reply_text("❌ Not enough data to draw a chart right now.")
            return
        current = get_gold_price()
        send_photo(png, f"📸 Gold {period} chart | now: ${current:,.2f}")
    except Exception as e:
        await update.message.reply_text(f"❌ Chart failed: {e}")


async def indicators_cmd(update, context):
    await update.message.reply_text("📊 Fetching REAL indicators...")
    snap = get_technicals_snapshot()
    if not snap:
        await update.message.reply_text(
            "⚠️ Indicators need a TWELVEDATA_API_KEY (free at twelvedata.com)."
        )
        return
    current = get_gold_price()
    take = ask_ai_or_softfail(
        f"Current gold price: ${current}\nReal indicator readings:\n{snap}\n\n"
        "In under 100 words: what do these indicators say about short-term "
        "momentum? Bullish/bearish/neutral and why. Use emojis. Not financial advice."
    )
    await update.message.reply_text(f"📊 REAL TECHNICALS (15min):\n\n{snap}\n\n🧠 AI READ:\n\n{take}")


async def analyze(update, context):
    await update.message.reply_text("🧠 Analyzing... (10-20s)")
    try:
        history = get_recent_history()[["Close", "High", "Low"]].round(2).to_string()
        current = get_gold_price()
        tech = get_technicals_snapshot()
        prompt = (
            "You are a gold market assistant for a retail trader. "
            f"Recent gold futures data:\n{history}\n\nCurrent price: ${current}\n\n"
        )
        if tech:
            prompt += f"REAL indicator readings:\n{tech}\n\n"
        prompt += (
            "SHORT analysis (max 150 words): trend, key levels, one-sentence outlook. "
            "Use emojis. Add a not-financial-advice reminder."
        )
        response_text = ask_ai_or_softfail(prompt)
        await update.message.reply_text(
            f"🥇 Gold: ${current:,.2f}\n\n📊 AI ANALYSIS 📊\n\n{response_text}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Analysis failed: {e}")


async def alert_cmd(update, context):
    arg = " ".join(context.args)
    target, direction = parse_alert_arg(arg)
    if target is None:
        await update.message.reply_text(
            "📈 Usage: /alert 4700  (or /alert 4700 above  /alert 4500 below)\n"
            "Direction is auto-detected from the current price if not given!"
        )
        return
    current = get_gold_price()
    if direction is None:
        direction = "above" if target > current else "below"
    alerts = load_alerts()
    alerts.append({
        "target": target,
        "direction": direction,
        "set_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price_then": current,
    })
    save_alerts(alerts)
    emoji = "🚀" if direction == "above" else "📉"
    await update.message.reply_text(
        f"✅ Alert set! {emoji}\n\n"
        f"🎯 Ping me when gold goes {direction} ${target:,.2f}\n"
        f"(now: ${current:,.2f})\n\n"
        "⏳ Checking every 5 min — I'll message you the moment it hits!"
    )


async def alerts_list(update, context):
    alerts = load_alerts()
    if not alerts:
        await update.message.reply_text("📈 No active alerts! Set one: /alert 4700")
        return
    current = get_gold_price()
    lines = [f"📈 ACTIVE ALERTS (gold now: ${current:,.2f}):\n"]
    for a in alerts:
        emoji = "🚀" if a["direction"] == "above" else "📉"
        lines.append(f"{emoji} ${a['target']:,.2f} ({a['direction']}) — set {a['set_at']}")
    await update.message.reply_text("\n".join(lines))


async def risk(update, context):
    args = context.args
    if len(args) != 4:
        await update.message.reply_text(
            "🧮 Usage: /risk <account$> <risk%> <entry> <stop>\n"
            "Example: /risk 10000 1 4606 4590\n"
            "= $10k account, risking 1%, entry 4606, stop 4590"
        )
        return
    try:
        account, risk_pct, entry, stop = [float(a.replace(",", "")) for a in args]
        if entry == stop:
            await update.message.reply_text("❌ Entry and stop can't be the same!")
            return
        risk_amount = account * (risk_pct / 100)
        stop_distance = abs(entry - stop)
        position_oz = risk_amount / stop_distance
        lots_xau = position_oz / 100
        await update.message.reply_text(
            f"🧮 POSITION SIZE CALCULATOR\n\n"
            f"💰 Account: ${account:,.0f}\n"
            f"⚠️ Risk: {risk_pct}% = ${risk_amount:,.0f}\n"
            f"📍 Entry: {entry} | 🛑 Stop: {stop}\n"
            f"📏 Stop distance: ${stop_distance:,.2f}\n\n"
            f"✅ Position: {position_oz:,.2f} oz\n"
            f"✅ ≈ {lots_xau:.3f} standard lots (XAUUSD)\n\n"
            f"💡 Each $1 price move = ${position_oz:,.2f} P&L\n"
            "Not financial advice — manage your risk! 🛡️"
        )
    except ValueError:
        await update.message.reply_text("❌ All 4 values must be numbers!")


async def live(update, context):
    question = " ".join(context.args) if context.args else "What is happening with gold prices today?"
    await update.message.reply_text("🌐 Searching the live web + thinking...")
    await update.message.chat.send_action("typing")
    answer = ask_gemini(
        f"You are a gold trading assistant. Search the web for CURRENT info and answer. "
        f"Under 200 words, use emojis, include today's key numbers if found. "
        f"Not financial advice.\n\nQuestion: {question}",
        use_search=True,
    )
    await update.message.reply_text(f"🌐 LIVE ANSWER:\n\n{answer or '😴 AI brains resting — try /news!'}")


async def news(update, context):
    await update.message.reply_text("📰 Fetching today's gold news...")
    try:
        headlines = fetch_gold_news()
        news_text = "\n".join(f"• {h}" for h in headlines)
        summary = ask_gemini(
            f"Today's gold headlines:\n{news_text}\n\n"
            "Under 150 words: today's key theme for gold traders. "
            "Use emojis. End with 'Not financial advice.'"
        )
        if not summary:
            await update.message.reply_text(f"📰 TODAY'S GOLD HEADLINES:\n\n{news_text}")
            return
        await update.message.reply_text(
            f"📰 TODAY'S GOLD HEADLINES:\n\n{news_text}\n\n🧠 AI TAKE:\n\n{summary}"
        )
    except Exception as e:
        await update.message.reply_text(f"📰 RSS failed ({e}). Trying live search...")
        answer = ask_gemini(
            "Summarize today's gold market news in under 150 words. "
            "Use emojis. Not financial advice.",
            use_search=True,
        )
        await update.message.reply_text(f"📰 NEWS (via live search):\n\n{answer or '😴 AI brains resting!'}")


async def calendar(update, context):
    await update.message.reply_text("📅 Fetching high-impact economic events...")
    try:
        events, source = get_events_resilient()
    except Exception:
        await update.message.reply_text(
            "⚠️ All calendar sources failed right now\n"
            "👉 https://www.forexfactory.com/calendar\n"
            "💡 Or ask me directly: /live what events today affect gold?"
        )
        return
    if source.startswith("cached"):
        await update.message.reply_text(f"⚠️ Live feed blocked — showing {source} calendar:")
    event_list = "\n".join(events)
    take = ask_gemini(
        f"Upcoming HIGH-impact economic events:\n{event_list}\n\n"
        "In under 120 words: which of these matter MOST for gold and why. "
        "Use emojis. Not financial advice."
    )
    text = f"📅 UPCOMING HIGH-IMPACT NEWS ({source}):\n\n{event_list}"
    if take:
        text += f"\n\n🧠 GOLD IMPACT:\n\n{take}"
    await update.message.reply_text(text)


async def note(update, context):
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text(
            "📓 Usage: /note LONG 4606 SL 4590 TP 4650 - breakout retest\n"
            "(SL/TP keywords are optional but make /stats smarter!)"
        )
        return
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text("⚠️ Journal needs GITHUB_TOKEN + GITHUB_REPO in environment!")
        return
    await update.message.reply_text("📓 Saving to journal...")
    try:
        fields = parse_note_fields(text)
        stamp = add_journal_entry(text)
        sha, stats = gh_read_json(STATS_PATH)
        stats["trades"].append({
            "note": text[:200],
            "opened": stamp,
            "fields": fields,
            "closed": None,
        })
        gh_write_json(STATS_PATH, stats, sha)
        struct = ""
        if fields:
            struct = "\n🔍 Detected: " + " | ".join(f"{k}={v}" for k, v in fields.items())
        await update.message.reply_text(
            f"✅ Saved to Journal/trades.md!\n\n🕒 {stamp}\n📝 {text}{struct}\n\n"
            "When you exit: /close +2R (or -1R) 📊"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Journal save failed: {e}")


async def close(update, context):
    """Usage: /close +2R optional comment   or   /close -1R stopped out"""
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text(
            "📊 Usage: /close <result in R>\n"
            "Examples:\n/close +2R hit TP cleanly\n/close -1R stopped out\n"
            "/close 0R scratched, flat exit"
        )
        return
    m = re.search(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*R\b", text.upper())
    if not m:
        await update.message.reply_text("❌ Include the R multiple, e.g. /close +2R or /close -1R")
        return
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text("⚠️ Stats need GITHUB_TOKEN + GITHUB_REPO in environment!")
        return
    r_mult = float(m.group(1))
    comment = text[m.end():].strip(" -–")
    try:
        sha, stats = gh_read_json(STATS_PATH)
        open_trades = [t for t in stats["trades"] if t.get("closed") is None]
        entry = {
            "closed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "r": r_mult,
            "comment": comment,
        }
        if open_trades:
            open_trades[-1]["closed"] = entry
            label = open_trades[-1].get("note", "trade")[:60]
        else:
            stats["trades"].append({"note": "(no matching open note)", "closed": entry})
            label = "standalone result"
        gh_write_json(STATS_PATH, stats, sha)
        emoji = "✅" if r_mult > 0 else ("🟡" if r_mult == 0 else "🛑")
        await update.message.reply_text(
            f"{emoji} Trade closed: {r_mult:+.1f}R\n📝 {label}\n"
            + (f"💬 {comment}\n" if comment else "")
            + "\n📊 See the scoreboard: /stats"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Close failed: {e}")


async def stats(update, context):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text("⚠️ Stats need GITHUB_TOKEN + GITHUB_REPO in environment!")
        return
    try:
        _, stats = gh_read_json(STATS_PATH)
    except Exception as e:
        await update.message.reply_text(f"❌ Couldn't read stats: {e}")
        return
    closed = [t["closed"] for t in stats["trades"] if t.get("closed")]
    if not closed:
        await update.message.reply_text(
            "📊 No closed trades yet!\n"
            "Workflow: /note open the trade → /close +2R when done."
        )
        return
    rs = [c["r"] for c in closed]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    win_rate = len(wins) / len(rs) * 100
    total_r = sum(rs)
    avg_r = total_r / len(rs)
    best = max(rs)
    worst = min(rs)
    streak, best_streak = 0, 0
    for r in rs:
        streak = streak + 1 if r > 0 else 0
        best_streak = max(best_streak, streak)
    bar_w, bar_l = "🟩" * len(wins), "🟥" * len(losses)
    verdict = ""
    if len(rs) >= 5:
        if total_r > 0 and win_rate >= 45:
            verdict = "💪 Positive expectancy — the system is working. Stay disciplined."
        elif total_r > 0:
            verdict = "🟢 Profitable but fragile — protect those wins, tighten entries."
        else:
            verdict = "⚠️ Negative expectancy — reduce size, review entries with /summary."
    await update.message.reply_text(
        f"📊 TRADING SCOREBOARD\n\n"
        f"🔒 Closed trades: {len(rs)}\n"
        f"🏆 Win rate: {win_rate:.0f}% ({len(wins)}W / {len(losses)}L"
        + (f" / {len(rs)-len(wins)-len(losses)}BE)" if len(rs) - len(wins) - len(losses) else ")")
        + f"\n💰 Total: {total_r:+.1f}R | Avg: {avg_r:+.2f}R per trade\n"
        f"🔥 Best streak: {best_streak} wins\n"
        f"⭐ Best: {best:+.1f}R | 💀 Worst: {worst:+.1f}R\n\n"
        f"{bar_w}{bar_l}\n"
        + (f"\n🧠 {verdict}" if verdict else "\n📈 Log 5+ closed trades for an AI verdict")
        + "\n\nNot financial advice 🛡️"
    )


async def trades(update, context):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text("⚠️ Journal not configured (GITHUB_TOKEN/GITHUB_REPO).")
        return
    try:
        blocks = get_recent_entries(5)
        if not blocks:
            await update.message.reply_text("📓 Journal is empty! Add your first /note 📝")
            return
        await update.message.reply_text("📓 LAST 5 JOURNAL ENTRIES:\n" + "\n\n".join(blocks))
    except Exception as e:
        await update.message.reply_text(f"❌ Couldn't read journal: {e}")


async def summary(update, context):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text("⚠️ Journal not configured (GITHUB_TOKEN/GITHUB_REPO).")
        return
    await update.message.reply_text("🧠 Reading your journal...")
    try:
        blocks = get_recent_entries(10)
        if not blocks:
            await update.message.reply_text("📓 Journal is empty - nothing to summarize yet!")
            return
        journal_text = "\n\n".join(blocks)
        take = ask_ai_or_softfail(
            f"A trader's recent gold trade journal:\n{journal_text}\n\n"
            "In under 150 words: patterns in their entries (biases, repeated setups, "
            "risk habits), one piece of constructive advice. Use emojis. "
            "Not financial advice."
        )
        await update.message.reply_text(f"🧠 JOURNAL REVIEW:\n\n{take}")
    except Exception as e:
        await update.message.reply_text(f"❌ Summary failed: {e}")


async def help_cmd(update, context):
    await update.message.reply_text(
        "Commands: /start /price /chart /analyze /indicators /alert /alerts /risk "
        "/note /close /stats /trades /summary /news /calendar /live /help\n\n"
        "💬 Or type any question normally!"
    )

# ============ FREE CHAT ============

GOLD_CONTEXT = """You are a friendly gold trading assistant chatting on Telegram.
You help analyze XAU/USD (gold). Be concise (under 150 words), use emojis,
give balanced views, and remind lightly that this is not financial advice.
Always reply only in English.
IMPORTANT: You have NO access to live prices, charts, indicators, or calendars.
NEVER invent price numbers, RSI values, levels, dates, or events.
If asked for current/live data, charts, or technical indicator readings, reply exactly:
"I can't see live charts - use /price for the live gold price, /chart for a chart image, /analyze for AI analysis of real recent data, /indicators for real RSI/MACD, or /live for real-time web search."
You MAY freely explain concepts, strategies, math, and general knowledge."""


async def free_chat(update, context):
    user_text = update.message.text
    print(f"💬 Chat received: {user_text}")
    await update.message.chat.send_action("typing")

    full_prompt = f"{GOLD_CONTEXT}\n\nUser: {user_text}"

    reply = ask_groq(full_prompt)
    if not reply:
        reply = ask_gemini(full_prompt)

    try:
        await update.message.reply_text(reply or "😴 All AI brains resting — try again later!")
        print("✅ Reply sent")
    except Exception as e:
        await update.message.reply_text(f"😵 Brain glitch: {e}")

# ============ RUN EVERYTHING ============

brain_status = "✅" if GROQ_API_KEY else "⚠️ add GROQ_API_KEY!"
td_status = "✅" if TWELVEDATA_KEY else "⚠️ Yahoo-only"
gh_status = "✅" if (GITHUB_TOKEN and GITHUB_REPO) else "⚠️ journal off"
print("🤖 Starting bot v6.1 + webhook server (port 5000)... press Ctrl+C to stop")
print(f"🧠 Dual-brain: Gemini(3 models) → Groq chain {brain_status}")
print(f"📡 Market data: TwelveData → Yahoo {td_status}")
print(f"📓 Journal + Stats → GitHub/{JOURNAL_DIR}: {gh_status} (repo: {GITHUB_REPO})")
print(f"⏰ Daily briefing: {BRIEFING_HOUR_UTC}:00 UTC")

threading.Thread(target=run_flask, daemon=True).start()
threading.Thread(target=check_alerts_loop, daemon=True).start()
threading.Thread(target=daily_briefing_loop, daemon=True).start()

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("price", price))
app.add_handler(CommandHandler("chart", chart))
app.add_handler(CommandHandler("analyze", analyze))
app.add_handler(CommandHandler("indicators", indicators_cmd))
app.add_handler(CommandHandler("alert", alert_cmd))
app.add_handler(CommandHandler("alerts", alerts_list))
app.add_handler(CommandHandler("risk", risk))
app.add_handler(CommandHandler("news", news))
app.add_handler(CommandHandler("calendar", calendar))
app.add_handler(CommandHandler("live", live))
app.add_handler(CommandHandler("note", note))
app.add_handler(CommandHandler("close", close))
app.add_handler(CommandHandler("stats", stats))
app.add_handler(CommandHandler("trades", trades))
app.add_handler(CommandHandler("summary", summary))
app.add_handler(CommandHandler("help", help_cmd))

# Must be LAST so slash commands still work first
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_chat))

app.run_polling()