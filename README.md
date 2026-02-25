# AI Quant Bot (1H Edition)

An institutional-grade algorithmic trading bot that scans the 1-Hour timeframe for Forex, Crypto, and Commodities. It uses Smart Money Concepts (SMC) including Order Blocks, Fair Value Gaps (FVG), and technical confluence to generate and broadcast high-probability trading signals.

## Features
* **Multi-Asset:** Supports Forex, Crypto, and Gold via the TwelveData API.
* **Smart Money Concepts:** Detects Institutional Order Blocks and FVGs.
* **Database Tracking:** Logs all signals to a local SQLite database (`trading_bot.db`).
* **Asynchronous Broadcasting:** Uses `python-telegram-bot` for non-blocking Telegram alerts.

## Configuration
Create a `.env` file in the root directory with the following variables:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=@YourChannelName
TD_API_KEY=your_twelvedata_api_key_here
