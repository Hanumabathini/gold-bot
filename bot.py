import os
import re
import threading
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types
from telegram.ext import Application, CommandHandler, MessageHandler, filters

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Gemini setup (legacy SDK - basic calls) ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-3.7-flash")  # ✅ YOUR WORKING MODEL

# --- NEW SDK client with GOOGLE SEARCH (live data!) ---
gclient = genai_new.Client(api_key=os.getenv("GEMINI_API_KEY"))
SEARCH_MODEL = "gemini-2.0-flash"  # search tool works on this; change if needed

SEARCH_TOOL = [genai_types.Tool(google_search=genai_types.GoogleSearch())]

def ask_live(prompt):
    """Ask Gemini WITH live Google Search access - knows today's news/prices."""
    try:
        resp = gclient.models.generate_content(
            model=SEARCH_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(tools=SEARCH_TOOL),
        )
        return resp.text
    except Exception as e:
        return f"(Live search unavailable: {e})"

# ================= HELPERS =================

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

def ask_gemini_signal(signal_type):
    try:
        prompt = (
            f"A trading indicator just fired a {signal_type} signal on gold futures. "
            f"Current price is ${get_gold_price()}. Recent data:\n{get_recent_history()}\n\n"
            f"In under 60 words, say whether recent price action supports this "
            f"{signal_type} entry. Start with AGREE or DISAGREE. Not financial advice."
        )
        return gemini.generate_content(prompt).text
    except Exception as e:
        return f"(Gemini unavailable: {e})"

# ================= FLASK WEBHOOK =================

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

    opinion = ask_gemini_signal(signal_type)
    send_message(f"🧠 GEMINI'S OPINION:\n\n{opinion}")

    return jsonify({"status": "ok"}), 200

@flask_app.route("/", methods=["GET"])
def home():
    return "Gold bot webhook is alive!", 200

def run_flask():
    flask_app.run(host="0.0.0.0", port=5000)

# ================= TELEGRAM COMMANDS =================

async def start(update, context):
    await update.message.reply_text(
        "👋 Gold Assistant online!\n\n"
        "/price - live price\n"
        "/analyze - AI analysis\n"
        "/news - today's gold headlines\n"
        "/calendar - ForexFactory high-impact events\n"
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
        response = gemini.generate_content(prompt)
        await update.message.reply_text(
            f"🥇 Gold: ${current:,.2f}\n\n📊 GEMINI ANALYSIS 📊\n\n{response.text}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Analysis failed: {e}")

# ---------- LIVE SEARCH: real-time news, prices, anything ----------

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
    """Today's gold headlines from Google News RSS (free, no key)."""
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
        summary = gemini.generate_content(prompt).text

        await update.message.reply_text(
            f"📰 TODAY'S GOLD HEADLINES:\n\n{news_text}\n\n"
            f"🧠 GEMINI'S TAKE:\n\n{summary}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ News failed: {e}")

# ---------- ECONOMIC CALENDAR (ForexFactory high-impact events) ----------

def fetch_ff_events():
    """Free ForexFactory weekly calendar - today's & upcoming high-impact only."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    r = requests.get(url, timeout=10)
    events = r.json()

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    out = []
    for ev in events:
        impact = str(ev.get("impact", "")).lower()
        if impact != "high":
            continue
        try:
            when = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
            if when >= now:
                local = when.strftime("%a %H:%M UTC")
                out.append(f"🔴 {local} | {ev['country']} | {ev['title']}")
        except Exception:
            continue
    return out[:12]

async def calendar(update, context):
    await update.message.reply_text("📅 Fetching ForexFactory high-impact events...")
    try:
        events = fetch_ff_events()
        if not events:
            await update.message.reply_text("✅ No more high-impact events this week!")
            return
        event_list = "\n".join(events)

        prompt = (
            f"Upcoming HIGH-impact economic events:\n{event_list}\n\n"
            "In under 120 words: which of these matter MOST for gold and why. "
            "Use emojis. Not financial advice."
        )
        take = gemini.generate_content(prompt).text

        await update.message.reply_text(
            f"📅 UPCOMING HIGH-IMPACT NEWS (ForexFactory):\n\n{event_list}\n\n"
            f"🧠 GOLD IMPACT:\n\n{take}"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Calendar failed: {e}")

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

# ================= FREE CHAT (normal text -> Gemini) =================

GOLD_CONTEXT = """You are a friendly gold trading assistant chatting on Telegram.
You help analyze XAU/USD (gold). Be concise (under 150 words), use emojis,
give balanced views, and remind lightly that this is not financial advice."""

async def free_chat(update, context):
    user_text = update.message.text
    print(f"💬 Chat received: {user_text}")
    await update.message.chat.send_action("typing")
    try:
        response = gemini.generate_content(f"{GOLD_CONTEXT}\n\nUser: {user_text}")
        await update.message.reply_text(response.text)
        print("✅ Reply sent")
    except Exception as e:
        await update.message.reply_text(f"😵 Brain glitch: {e}")

# ================= RUN EVERYTHING =================

print("🤖 Starting bot + webhook server (port 5000)... press Ctrl+C to stop")
print("💬 Free chat | 📰 /news | 📅 /calendar | 🌐 /live search — ALL READY")

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
