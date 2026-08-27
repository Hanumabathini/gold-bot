import os
import re
import threading
import requests
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

# ============ AI BRAINS: Gemini chain + Groq fallback ============

genai.configure(api_key=GEMINI_API_KEY)

MODEL_CHAIN = ["gemini-3.7-flash", "gemini-2.0-flash", "gemini-flash-latest"]

# Groq models - verified available for this key (best first)
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

# Build thinking-tags dynamically (survives any copy-paste mangling)
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


def clean_thinking(text):
    """Remove reasoning-model 'thinking' output leakage."""
    text = re.sub(
        THINK_OPEN + r".*?" + THINK_CLOSE, "", text, flags=re.DOTALL
    ).strip()
    # Unclosed thinking block: cut everything from its start
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

# ============ HELPERS ============

def get_gold_price():
    gold = yf.Ticker("GC=F")
    data = gold.history(period="1d")
    return round(data["Close"].iloc[-1], 2)


def get_recent_history():
    gold = yf.Ticker("GC=F")
    data = gold.history(period="5d")
    return data[["Close", "High", "Low"]].round(2).to_string()


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
        "/price - live price\n"
        "/analyze - AI analysis\n"
        "/news - today's gold headlines\n"
        "/calendar - high-impact economic events\n"
        "/live - ask AI anything with REAL-TIME web search\n"
        "/webhookstatus - check webhook\n"
        "/help - commands\n\n"
        "💬 Or just chat with me about gold!"
    )


async def price(update, context):
    await update.message.reply_text("⏳ Fetching...")
    try:
        await update.message.reply_text(f"🥇 Gold: ${get_gold_price():,.2f}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def analyze(update, context):
    await update.message.reply_text("🧠 Analyzing... (10-20s)")
    try:
        history = get_recent_history()
        current = get_gold_price()
        prompt = (
            "You are a gold market assistant for a retail trader. "
            f"Recent gold futures data:\n{history}\n\nCurrent price: ${current}\n\n"
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

# ---------- NEWS COMMAND (Google News RSS) ----------

def fetch_gold_news():
    """Today's gold headlines from Google News RSS (free)."""
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
            f"📰 TODAY'S GOLD HEADLINES:\n\n{news_text}\n\n"
            f"🧠 AI TAKE:\n\n{summary}"
        )
    except Exception as e:
        await update.message.reply_text(f"📰 RSS failed ({e}). Trying live search...")
        answer = ask_live(
            "Summarize today's gold market news in under 150 words. "
            "Use emojis. Not financial advice."
        )
        await update.message.reply_text(f"📰 NEWS (via live search):\n\n{answer}")

# ---------- ECONOMIC CALENDAR (ForexFactory + fallback) ----------

def fetch_ff_events():
    """ForexFactory weekly calendar - high-impact only."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=15)
    events = r.json()

    from datetime import datetime, timezone
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
        # FF fetch failed - do NOT let a non-search AI invent events!
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


async def webhookstatus(update, context):
    await update.message.reply_text(
        "🔗 Webhook server runs on port 5000.\n"
        "TradingView signals arrive instantly when everything is running!"
    )


async def help_cmd(update, context):
    await update.message.reply_text(
        "Commands: /start /price /analyze /news /calendar /live /webhookstatus /help\n\n"
        "💬 Or type any question normally!"
    )

# ============ FREE CHAT (Groq FIRST to save Gemini quota!) ============

GOLD_CONTEXT = """You are a friendly gold trading assistant chatting on Telegram.
You help analyze XAU/USD (gold). Be concise (under 150 words), use emojis,
give balanced views, and remind lightly that this is not financial advice.
Always reply only in English.
IMPORTANT: You have NO access to live prices, charts, indicators, or calendars.
NEVER invent price numbers, RSI values, levels, dates, or events.
If asked for current/live data, charts, or technical indicator readings, reply exactly:
"I can't see live charts - use /price for the live gold price, /analyze for AI analysis of real recent data, or /live for real-time web search."
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
print("🤖 Starting bot + webhook server (port 5000)... press Ctrl+C to stop")
print(f"🧠 Dual-brain: Gemini(3 models) → Groq hard-coded chain {brain_status}")

threading.Thread(target=run_flask, daemon=True).start()

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("price", price))
app.add_handler(CommandHandler("analyze", analyze))
app.add_handler(CommandHandler("news", news))
app.add_handler(CommandHandler("calendar", calendar))
app.add_handler(CommandHandler("live", live))
app.add_handler(CommandHandler("webhookstatus", webhookstatus))
app.add_handler(CommandHandler("help", help_cmd))

# Must be LAST so slash commands still work first
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_chat))

app.run_polling()
