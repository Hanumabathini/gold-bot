import os
import re
import json
import base64
import threading
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai
from telegram.ext import Application, CommandHandler, MessageHandler, filters

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. Hanumabathini/trading

# ============ AI BRAINS: Gemini chain + Groq fallback ============

genai.configure(api_key=GEMINI_API_KEY)

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
    """Remove reasoning-model 'thinking' output leakage."""
    text = re.sub(THINK_OPEN + r".*?" + THINK_CLOSE, "", text, flags=re.DOTALL).strip()
    if THINK_OPEN in text:
        text = text.split(THINK_OPEN)[0].strip()
    return text


def ask_groq(prompt):
    """FREE fallback brain via Groq - tries known-good models in order."""
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


def ask_gemini(prompt, tools=None):
    """Try Gemini models in order (skips out-of-quota), then Groq."""
    prompt = "Reply ONLY in English.\n\n" + prompt
    for model_name in MODEL_CHAIN:
        try:
            model = genai.GenerativeModel(model_name)
            if tools:
                resp = model.generate_content(prompt, tools=tools)
            else:
                resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "429" in err or "exceeded" in err:
                print(f"⚠️ {model_name} out of quota, trying next...")
                continue
            print(f"Gemini {model_name} error: {e}")
            continue
    groq_answer = ask_groq(prompt)
    if groq_answer:
        return groq_answer
    return "😴 All AI brains are resting (daily quotas used). Try again tomorrow!"


def ask_live(prompt):
    """AI WITH Google Search grounding (Gemini only - Groq can't search)."""
    try:
        return ask_gemini(prompt, tools="google_search_retrieval")
    except Exception as e:
        return f"(Live search unavailable: {e})"

# ============ MARKET DATA: TwelveData first, Yahoo fallback ============

TD_BASE = "https://api.twelvedata.com"


def get_gold_price():
    """Real XAU/USD spot via TwelveData; falls back to Yahoo futures."""
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


def get_recent_history():
    gold = yf.Ticker("GC=F")
    data = gold.history(period="5d")
    return data[["Close", "High", "Low"]].round(2).to_string()


def get_indicator(indicator, interval="15min"):
    """Real technical indicator from TwelveData (RSI, MACD, EMA...)."""
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
    """Compact text block of REAL indicators for the AI (None-safe)."""
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

# ============ TRADE JOURNAL -> GITHUB (Obsidian bridge) ============

GH_API = "https://api.github.com"
JOURNAL_DIR = "Journal"          # folder inside the repo
JOURNAL_FILE = "trades.md"       # single running journal file


def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def gh_get_file():
    """Return (sha, decoded_text). sha=None if file doesn't exist yet."""
    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{JOURNAL_DIR}/{JOURNAL_FILE}"
    r = requests.get(url, headers=gh_headers(), timeout=15)
    if r.status_code == 200:
        j = r.json()
        return j["sha"], base64.b64decode(j["content"]).decode("utf-8")
    if r.status_code == 404:
        return None, ""
    raise Exception(f"GitHub read failed {r.status_code}: {r.text[:150]}")


def gh_write_file(new_content, sha):
    """Create/update the journal file and commit."""
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
        raise Exception(f"GitHub write failed {r.status_code}: {r.text[:200]}")


def add_journal_entry(text):
    """Append a timestamped note to the journal file. Returns commit message."""
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
    """Extract the last N '## ' entry headers + their lines."""
    _, content = gh_get_file()
    if not content:
        return None
    blocks = content.split("\n## ")
    blocks = [b if b.startswith("## ") else "## " + b for b in blocks if b.strip()]
    return blocks[-n:] if blocks else None

# ============ TELEGRAM SEND ============

def send_message(text):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"Send failed: {e}")

# ============ FLASK WEBHOOK ============

flask_app = Flask(__name__)


@flask_app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    signal_type = data.get("signal", "SIGNAL").upper()
    tv_message = data.get("message", "TradingView alert")

    send_message(
        f"⚡ TRADINGVIEW SIGNAL ⚡\n\n"
        f"🚨 {signal_type} on Gold!\n"
        f"📄 {tv_message}"
    )

    prompt = (
        f"A trading indicator just fired a {signal_type} signal on gold futures. "
        f"Current price is ${get_gold_price()}. Recent data:\n{get_recent_history()}\n\n"
        f"In under 60 words, say whether recent price action supports this "
        f"{signal_type} entry. Start with AGREE or DISAGREE. Not financial advice."
    )
    opinion = ask_gemini(prompt)
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
        "👋 Gold Assistant online!\n\n"
        "/price - live XAU/USD price\n"
        "/analyze - AI analysis with REAL indicators\n"
        "/indicators - RSI / EMA / MACD snapshot\n"
        "/news - today's gold headlines\n"
        "/calendar - high-impact economic events\n"
        "/live - ask AI anything with REAL-TIME web search\n"
        "/note <text> - log to your Obsidian journal 📓\n"
        "/trades - last 5 journal entries\n"
        "/summary - AI reviews your recent journal\n"
        "/help - commands\n\n"
        "💬 Or just chat with me about gold!"
    )


async def price(update, context):
    await update.message.reply_text("⏳ Fetching...")
    try:
        source = "TwelveData (spot)" if TWELVEDATA_KEY else "Yahoo (GC=F)"
        await update.message.reply_text(f"🥇 Gold: ${get_gold_price():,.2f}\n📡 Source: {source}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def indicators_cmd(update, context):
    await update.message.reply_text("📊 Fetching REAL indicators...")
    snap = get_technicals_snapshot()
    if not snap:
        await update.message.reply_text(
            "⚠️ Indicators need a TWELVEDATA_API_KEY (free at twelvedata.com)."
        )
        return
    current = get_gold_price()
    take = ask_gemini(
        f"Current gold price: ${current}\nReal indicator readings:\n{snap}\n\n"
        "In under 100 words: what do these indicators say about short-term "
        "momentum? Bullish/bearish/neutral and why. Use emojis. Not financial advice."
    )
    await update.message.reply_text(f"📊 REAL TECHNICALS (15min):\n\n{snap}\n\n🧠 AI READ:\n\n{take}")


async def analyze(update, context):
    await update.message.reply_text("🧠 Analyzing... (10-20s)")
    try:
        history = get_recent_history()
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
        response_text = ask_gemini(prompt)
        await update.message.reply_text(
            f"🥇 Gold: ${current:,.2f}\n\n📊 AI ANALYSIS 📊\n\n{response_text}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Analysis failed: {e}")

# ---------- LIVE SEARCH ----------

async def live(update, context):
    question = " ".join(context.args) if context.args else "What is happening with gold prices today?"

    await update.message.reply_text("🌐 Searching the live web + thinking...")
    await update.message.chat.send_action("typing")

    answer = ask_live(
        f"You are a gold trading assistant. Search the web for CURRENT info and answer. "
        f"Under 200 words, use emojis, include today's key numbers if found. "
        f"Not financial advice.\n\nQuestion: {question}"
    )
    await update.message.reply_text(f"🌐 LIVE ANSWER:\n\n{answer}")

# ---------- NEWS COMMAND ----------

def fetch_gold_news():
    url = "https://news.google.com/rss/search?q=gold+price+when:1d&hl=en-US&gl=US&ceid=US:en"
    r = requests.get(url, timeout=10)
    titles = re.findall(r"<title>(.*?)</title>", r.text)[1:8]
    return [t.replace("&amp;", "&").replace("&#39;", "'") for t in titles]


async def news(update, context):
    await update.message.reply_text("📰 Fetching today's gold news...")
    try:
        headlines = fetch_gold_news()
        news_text = "\n".join(f"• {h}" for h in headlines)
        prompt = (
            f"Today's gold headlines:\n{news_text}\n\n"
            "Under 150 words: today's key theme for gold traders. "
            "Use emojis. End with 'Not financial advice.'"
        )
        summary = ask_gemini(prompt)
        await update.message.reply_text(
            f"📰 TODAY'S GOLD HEADLINES:\n\n{news_text}\n\n🧠 AI TAKE:\n\n{summary}"
        )
    except Exception as e:
        await update.message.reply_text(f"📰 RSS failed ({e}). Trying live search...")
        answer = ask_live(
            "Summarize today's gold market news in under 150 words. "
            "Use emojis. Not financial advice."
        )
        await update.message.reply_text(f"📰 NEWS (via live search):\n\n{answer}")

# ---------- ECONOMIC CALENDAR ----------

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
    return out[:12]


async def calendar(update, context):
    await update.message.reply_text("📅 Fetching high-impact economic events...")
    try:
        events = fetch_ff_events()
    except Exception:
        await update.message.reply_text(
            "⚠️ Couldn't fetch the ForexFactory feed right now.\n"
            "👉 Check high-impact events here:\n"
            "https://www.forexfactory.com/calendar\n\n"
            "💡 Tip: try again later, or use /live for web-searched answers."
        )
        return

    if not events:
        await update.message.reply_text("✅ No more high-impact events this week!")
        return
    event_list = "\n".join(events)
    prompt = (
        f"Upcoming HIGH-impact economic events:\n{event_list}\n\n"
        "In under 120 words: which of these matter MOST for gold and why. "
        "Use emojis. Not financial advice."
    )
    take = ask_gemini(prompt)
    await update.message.reply_text(
        f"📅 UPCOMING HIGH-IMPACT NEWS (ForexFactory):\n\n{event_list}\n\n🧠 GOLD IMPACT:\n\n{take}"
    )

# ---------- JOURNAL COMMANDS ----------

async def note(update, context):
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text(
            "📓 Usage: /note LONG 4650 SL 4630 TP 4700 - breakout retest\n"
            "Anything you write is saved to your GitHub journal (appears in Obsidian)!"
        )
        return
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text(
            "⚠️ Journal needs GITHUB_TOKEN + GITHUB_REPO in environment!"
        )
        return
    await update.message.reply_text("📓 Saving to journal...")
    try:
        stamp = add_journal_entry(text)
        await update.message.reply_text(
            f"✅ Saved to Journal/trades.md!\n\n🕒 {stamp}\n📝 {text}\n\n"
            "→ It's committed to GitHub now. Pull in Obsidian to see it! 🎉"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Journal save failed: {e}")


async def trades(update, context):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        await update.message.reply_text("⚠️ Journal not configured (GITHUB_TOKEN/GITHUB_REPO).")
        return
    try:
        blocks = get_recent_entries(5)
        if not blocks:
            await update.message.reply_text("📓 Journal is empty! Add your first /note 📝")
            return
        await update.message.reply_text(
            "📓 LAST 5 JOURNAL ENTRIES:\n" + "\n\n".join(blocks)
        )
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
        take = ask_gemini(
            f"A trader's recent gold trade journal:\n{journal_text}\n\n"
            "In under 150 words: patterns in their entries (biases, repeated setups, "
            "risk habits), one piece of constructive advice. Use emojis. "
            "Not financial advice."
        )
        await update.message.reply_text(f"🧠 JOURNAL REVIEW:\n\n{take}")
    except Exception as e:
        await update.message.reply_text(f"❌ Summary failed: {e}")


async def webhookstatus(update, context):
    await update.message.reply_text(
        "🔗 Webhook server runs on port 5000.\n"
        "TradingView alerts POST to: https://YOUR-RENDER-APP.onrender.com/webhook"
    )


async def help_cmd(update, context):
    await update.message.reply_text(
        "Commands: /start /price /analyze /indicators /news /calendar /live "
        "/note /trades /summary /webhookstatus /help\n\n"
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
"I can't see live charts - use /price for the live gold price, /analyze for AI analysis of real recent data, /indicators for real RSI/MACD, or /live for real-time web search."
You MAY freely explain concepts, strategies, indicator math, and general knowledge."""


async def free_chat(update, context):
    user_text = update.message.text
    print(f"💬 Chat received: {user_text}")
    await update.message.chat.send_action("typing")

    full_prompt = f"{GOLD_CONTEXT}\n\nUser: {user_text}"

    reply = ask_groq(full_prompt)
    if not reply:
        reply = ask_gemini(full_prompt)

    try:
        await update.message.reply_text(reply)
        print("✅ Reply sent")
    except Exception as e:
        await update.message.reply_text(f"😵 Brain glitch: {e}")

# ============ RUN EVERYTHING ============

brain_status = "✅" if GROQ_API_KEY else "⚠️ add GROQ_API_KEY!"
td_status = "✅" if TWELVEDATA_KEY else "⚠️ Yahoo-only"
gh_status = "✅" if (GITHUB_TOKEN and GITHUB_REPO) else "⚠️ journal off"
print("🤖 Starting bot + webhook server (port 5000)... press Ctrl+C to stop")
print(f"🧠 Dual-brain: Gemini(3 models) → Groq chain {brain_status}")
print(f"📡 Market data: TwelveData → Yahoo {td_status}")
print(f"📓 Journal → GitHub/{JOURNAL_DIR}: {gh_status}")

threading.Thread(target=run_flask, daemon=True).start()

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("price", price))
app.add_handler(CommandHandler("analyze", analyze))
app.add_handler(CommandHandler("indicators", indicators_cmd))
app.add_handler(CommandHandler("news", news))
app.add_handler(CommandHandler("calendar", calendar))
app.add_handler(CommandHandler("live", live))
app.add_handler(CommandHandler("note", note))
app.add_handler(CommandHandler("trades", trades))
app.add_handler(CommandHandler("summary", summary))
app.add_handler(CommandHandler("webhookstatus", webhookstatus))
app.add_handler(CommandHandler("help", help_cmd))

# Must be LAST so slash commands still work first
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_chat))

app.run_polling()
