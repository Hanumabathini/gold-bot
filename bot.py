import os
import threading
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import yfinance as yf
import google.generativeai as genai
from telegram.ext import Application, CommandHandler

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Gemini setup ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-3.6-flash")

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

    # Instant relay to Telegram
    send_message(
        f"⚡ TRADINGVIEW SIGNAL ⚡\n\n"
        f"🚨 {signal_type} on Gold!\n"
        f"📄 {tv_message}"
    )

    # Gemini's second opinion
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
        "/webhookstatus - check webhook\n"
        "/help - commands"
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

async def webhookstatus(update, context):
    await update.message.reply_text(
        "🔗 Webhook server runs on port 5000.\n"
        "Cloudflared terminal must stay open with the tunnel.\n"
        "TradingView signals arrive instantly when everything is running!"
    )

async def help_cmd(update, context):
    await update.message.reply_text("Commands: /start /price /analyze /webhookstatus /help")

# ================= RUN EVERYTHING =================

print("🤖 Starting bot + webhook server (port 5000)... press Ctrl+C to stop")

threading.Thread(target=run_flask, daemon=True).start()

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("price", price))
app.add_handler(CommandHandler("analyze", analyze))
app.add_handler(CommandHandler("webhookstatus", webhookstatus))
app.add_handler(CommandHandler("help", help_cmd))
app.run_polling()
