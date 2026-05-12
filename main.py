import os
import json
import time
import base64
import tempfile
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import yfinance as yf

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

TEXT_MODEL_NAME = "deepseek/deepseek-chat"

# 如果图片模型不可用，可换成：
# "google/gemini-2.0-flash-001"
# "openai/gpt-4.1-mini"
VISION_MODEL_NAME = "google/gemini-2.0-flash-001"


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "15m"
LIMIT = 180

MEMORY_FILE = "memory.json"
ALERT_FILE = "alert_users.json"
CHINESE_NEWS_CACHE_FILE = "chinese_news_cache.json"
MACRO_CACHE_FILE = "macro_cache.json"

ALERT_COOLDOWN_SECONDS = 1800
MACRO_ALERT_COOLDOWN_SECONDS = 3600
BREAKING_NEWS_COOLDOWN_SECONDS = 900
BREAKING_NEWS_STATE_FILE = "breaking_news_state.json"

MULTI_TIMEFRAMES = ["15m", "1h", "4h"]


SYMBOL_MAP = {
    # Crypto
    "btc": "BTCUSDT",
    "bitcoin": "BTCUSDT",
    "比特币": "BTCUSDT",
    "大饼": "BTCUSDT",

    "eth": "ETHUSDT",
    "ethereum": "ETHUSDT",
    "以太坊": "ETHUSDT",
    "姨太": "ETHUSDT",

    "sol": "SOLUSDT",
    "solana": "SOLUSDT",
    "bnb": "BNBUSDT",
    "xrp": "XRPUSDT",
    "doge": "DOGEUSDT",
    "狗狗币": "DOGEUSDT",
    "ada": "ADAUSDT",
    "avax": "AVAXUSDT",
    "link": "LINKUSDT",
    "dot": "DOTUSDT",
    "ltc": "LTCUSDT",
    "etc": "ETCUSDT",

    # Gold / Silver
    "黄金": "GC=F",
    "金价": "GC=F",
    "现货黄金": "GC=F",
    "伦敦金": "GC=F",
    "gold": "GC=F",
    "xau": "GC=F",
    "xauusd": "GC=F",

    "白银": "SI=F",
    "银价": "SI=F",
    "现货白银": "SI=F",
    "silver": "SI=F",
    "xag": "SI=F",
    "xagusd": "SI=F",

    # Forex
    "eurusd": "EURUSD=X",
    "欧元": "EURUSD=X",
    "欧美": "EURUSD=X",
    "欧元美元": "EURUSD=X",

    "gbpusd": "GBPUSD=X",
    "英镑": "GBPUSD=X",
    "镑美": "GBPUSD=X",
    "英镑美元": "GBPUSD=X",

    "usdjpy": "JPY=X",
    "日元": "JPY=X",
    "美日": "JPY=X",
    "美元日元": "JPY=X",

    "audusd": "AUDUSD=X",
    "澳元": "AUDUSD=X",
    "澳美": "AUDUSD=X",
    "澳元美元": "AUDUSD=X",

    "usdcad": "CAD=X",
    "加元": "CAD=X",
    "美加": "CAD=X",
    "美元加元": "CAD=X",

    "usdchf": "CHF=X",
    "瑞郎": "CHF=X",
    "美瑞": "CHF=X",
    "美元瑞郎": "CHF=X",

    "nzdusd": "NZDUSD=X",
    "纽元": "NZDUSD=X",
    "纽美": "NZDUSD=X",
    "纽元美元": "NZDUSD=X",
}


INTERVAL_MAP = {
    "15m": "15m",
    "15分钟": "15m",
    "15分": "15m",

    "1h": "1h",
    "1小时": "1h",
    "一小时": "1h",
    "小时": "1h",

    "4h": "4h",
    "4小时": "4h",
    "四小时": "4h",

    "1d": "1d",
    "日线": "1d",
    "一天": "1d",
    "daily": "1d",
}


TRADING_PLAN_KEYWORDS = [
    "怎么做", "如何做", "交易计划", "计划", "策略", "进场计划",
    "做单", "布局", "怎么操作", "给我计划", "计划a", "plan",
    "btc怎么做", "黄金怎么做", "eth怎么做"
]


MARKET_OVERVIEW_KEYWORDS = [
    "现在市场怎样", "市场怎样", "市场怎么样", "今天市场", "整体市场",
    "市场情绪", "现在行情整体", "今天适合交易吗", "今天危险吗",
    "今晚危险吗", "现在适合交易吗", "风险大吗", "市场总览"
]

WHY_MOVE_KEYWORDS = [
    "为什么涨", "为什么跌", "为什么拉", "为什么跳水", "突然拉",
    "突然跌", "突然涨", "什么原因", "为啥涨", "为啥跌",
    "是不是消息", "是不是新闻", "为什么黄金跌", "为什么btc涨"
]

TEACHING_KEYWORDS = [
    "什么是", "什么意思", "解释一下", "教学", "怎么理解",
    "bos是什么", "choch是什么", "fvg是什么", "ob是什么", "流动性是什么",
    "如何看", "怎么判断"
]

RISK_MODE_KEYWORDS = [
    "能不能追", "该不该追", "危险吗", "风险大吗", "会不会被套",
    "适合交易吗", "要不要等", "能不能进", "现在进安全吗"
]

BREAKING_NEWS_KEYWORDS = [
    "突发", "快讯", "美联储", "鲍威尔", "cpi", "非农", "初请",
    "利率决议", "fomc", "降息", "加息", "通胀", "战争", "袭击",
    "爆炸", "制裁", "etf", "比特币", "btc", "黄金", "美元",
    "美债", "暴涨", "暴跌", "跳水", "拉升", "避险"
]


HIGH_IMPACT_KEYWORDS = [
    "cpi", "consumer price", "inflation", "pce", "core pce",
    "non farm", "nonfarm", "nfp", "payroll", "employment change",
    "fomc", "fed", "federal reserve", "powell",
    "interest rate", "rate decision", "rate statement",
    "unemployment", "jobless", "employment",
    "gdp", "retail sales", "ism", "pmi",
    "ppi", "producer price",
    "ecb", "boj", "boe",
]

CHINESE_HIGH_IMPACT_KEYWORDS = [
    "cpi", "非农", "美联储", "鲍威尔", "fomc", "利率决议", "降息", "加息",
    "通胀", "pce", "ppi", "gdp", "失业率", "初请", "就业", "零售销售",
    "美元指数", "美债收益率", "黄金", "原油", "比特币", "btc", "以太坊",
    "etf", "战争", "地缘", "制裁", "央行", "欧洲央行", "日本央行", "英国央行"
]

MACRO_EVENT_ALIASES = {
    "nfp": ["non-farm", "nonfarm", "payroll", "非农"],
    "cpi": ["cpi", "consumer price", "通胀", "消费者物价"],
    "jobless": ["jobless", "initial claims", "初请", "失业金"],
    "fomc": ["fomc", "federal reserve", "fed interest", "利率决议", "美联储"],
    "pce": ["pce", "personal consumption", "核心pce"],
    "ppi": ["ppi", "producer price", "生产者物价"],
    "gdp": ["gdp", "国内生产总值"],
    "retail": ["retail sales", "零售销售"],
    "pmi": ["pmi", "ism"],
}


MACRO_TRANSLATION = {
    "Non-Farm Employment Change": "美国非农就业人数",
    "Non-Farm Payrolls": "美国非农就业人数",
    "Average Hourly Earnings m/m": "美国平均时薪月率",
    "Unemployment Rate": "美国失业率",
    "Initial Jobless Claims": "美国初请失业金人数",
    "Continuing Jobless Claims": "美国续请失业金人数",
    "CPI m/m": "美国 CPI 月率",
    "CPI y/y": "美国 CPI 年率",
    "Core CPI m/m": "美国核心 CPI 月率",
    "Core CPI y/y": "美国核心 CPI 年率",
    "PPI m/m": "美国 PPI 月率",
    "Core PPI m/m": "美国核心 PPI 月率",
    "Core PCE Price Index m/m": "美国核心 PCE 物价指数月率",
    "Federal Funds Rate": "美联储利率决议",
    "FOMC Statement": "FOMC 政策声明",
    "FOMC Press Conference": "美联储新闻发布会",
    "Fed Chair Powell Speaks": "美联储主席鲍威尔讲话",
    "Advance GDP q/q": "美国 GDP 季率初值",
    "GDP q/q": "美国 GDP 季率",
    "Retail Sales m/m": "美国零售销售月率",
    "Core Retail Sales m/m": "美国核心零售销售月率",
    "ISM Manufacturing PMI": "美国 ISM 制造业 PMI",
    "ISM Services PMI": "美国 ISM 服务业 PMI",
    "Flash Manufacturing PMI": "制造业 PMI 初值",
    "Flash Services PMI": "服务业 PMI 初值",
    "Consumer Confidence": "消费者信心指数",
    "Crude Oil Inventories": "美国原油库存",
}

COUNTRY_TRANSLATION = {
    "USD": "美国",
    "EUR": "欧元区",
    "GBP": "英国",
    "JPY": "日本",
    "AUD": "澳洲",
    "CAD": "加拿大",
    "CHF": "瑞士",
    "NZD": "新西兰",
    "CNY": "中国",
    "United States": "美国",
    "Euro Zone": "欧元区",
    "United Kingdom": "英国",
    "Japan": "日本",
    "Australia": "澳洲",
    "Canada": "加拿大",
    "Switzerland": "瑞士",
    "New Zealand": "新西兰",
    "China": "中国",
}


SYMBOL_NEWS_KEYWORDS = {
    "GC=F": ["黄金", "金价", "美元", "美债", "通胀", "美联储", "cpi", "pce", "避险"],
    "SI=F": ["白银", "银价", "黄金", "美元", "美债", "通胀"],
    "BTCUSDT": ["比特币", "btc", "加密", "crypto", "etf", "美联储", "美元", "风险资产"],
    "ETHUSDT": ["以太坊", "eth", "加密", "crypto", "etf", "美联储", "风险资产"],
    "SOLUSDT": ["sol", "solana", "加密", "crypto", "风险资产"],
    "EURUSD=X": ["欧元", "欧洲央行", "ecb", "美元", "美联储"],
    "GBPUSD=X": ["英镑", "英国央行", "boe", "美元", "美联储"],
    "JPY=X": ["日元", "日本央行", "boj", "美元", "美联储", "美债"],
    "AUDUSD=X": ["澳元", "澳洲联储", "美元", "大宗商品"],
    "CAD=X": ["加元", "加拿大央行", "原油", "美元"],
    "CHF=X": ["瑞郎", "避险", "瑞士央行", "美元"],
    "NZDUSD=X": ["纽元", "新西兰联储", "美元"],
}

USD_SENSITIVE_SYMBOLS = [
    "GC=F", "SI=F",
    "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "CAD=X", "CHF=X", "NZDUSD=X",
    "BTCUSDT", "ETHUSDT", "SOLUSDT"
]


SYSTEM_PROMPT = """
你是一名有经验的交易员型 AI 行情助手。

你的身份：
- 你不是喊单老师
- 你不是投资顾问
- 你不保证收益
- 你只做行情分析、风险提醒和参考建议

你的回复风格：
- 像真人交易员聊天
- 简短、有判断、有经验感
- 不要像 AI 报告
- 不要太正式
- 不要机械列指标
- 重点讲：趋势、关键位、风险、怎么等确认

如果有重要经济数据或突发事件：
- 必须优先提醒新闻风险
- 不建议数据公布前重仓进场
- 建议等数据公布后 5~15 分钟，方向稳定再看
- 如果有 实际值 / 市场预测 / 前值，必须说明数据差异和可能影响

你禁止：
- 说肯定涨、一定跌、稳赚
- 叫用户满仓、重仓、梭哈
- 代替用户做最终交易决定
- 给出绝对确定性建议

每次回复最后都要自然带一句：
“以上仅供行情参考，不构成投资建议。”
"""


VISION_PROMPT = """
你是一名专业交易员型 AI 图表分析助手，擅长 TradingView / MT4 / MT5 K线图分析。

用户发来的是 K 线图、TradingView 截图或行情截图。

你要尽量识别：
1. 当前趋势：偏多、偏空、震荡
2. 关键支撑区
3. 关键压力区
4. 是否适合追多/追空
5. 更适合等待什么确认
6. BOS / CHOCH / OB / FVG / liquidity sweep / 假突破 / 供需区
7. 如果图上数字不清楚，要说明只能做结构判断

如果宏观层提示有 CPI、非农、美联储、初请等重要数据，要优先提醒消息面风险。

最后必须自然加一句：
“以上仅供行情参考，不构成投资建议。”
"""


def validate_env():
    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")

    if missing:
        raise RuntimeError("缺少环境变量：" + ", ".join(missing) + "。请检查 .env 文件。")


def is_crypto_symbol(symbol):
    return symbol.endswith("USDT")


def is_gold_symbol(symbol):
    return symbol == "GC=F"


def is_silver_symbol(symbol):
    return symbol == "SI=F"


def is_forex_symbol(symbol):
    return symbol.endswith("=X")


def get_asset_name(symbol):
    names = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "SOLUSDT": "SOL",
        "GC=F": "黄金",
        "SI=F": "白银",
        "EURUSD=X": "EURUSD 欧元美元",
        "GBPUSD=X": "GBPUSD 英镑美元",
        "JPY=X": "USDJPY 美元日元",
        "AUDUSD=X": "AUDUSD 澳元美元",
        "CAD=X": "USDCAD 美元加元",
        "CHF=X": "USDCHF 美元瑞郎",
        "NZDUSD=X": "NZDUSD 纽元美元",
    }
    return names.get(symbol, symbol)


def get_asset_macro_note(symbol):
    if is_gold_symbol(symbol) or is_silver_symbol(symbol):
        return "这是贵金属品种，除了技术面，更要重点看美元指数、美债收益率、CPI、PCE、非农和美联储讲话。数据前后不建议重仓。"

    if is_forex_symbol(symbol):
        return "这是外汇品种，除了技术面，更要重点看美元、美债、央行利率决议、CPI、就业数据和对应国家央行讲话。"

    if is_crypto_symbol(symbol):
        return "这是加密货币品种，除了技术面，也要注意美元、美债、ETF、监管消息和风险资产情绪。"

    return "需要同时结合技术面和消息面，不建议单看指标进场。"


def load_json(file_path, default_value):
    try:
        if not file_path:
            return default_value

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return default_value


def save_json(file_path, data):
    if not file_path:
        return

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def strip_html(raw_html):
    text = re.sub(r"<script[\s\S]*?</script>", "", raw_html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def detect_symbol(user_message, user_memory=None):
    text = user_message.lower()

    for key, symbol in SYMBOL_MAP.items():
        if key in text:
            return symbol

    if user_memory and user_memory.get("favorite_symbol"):
        return user_memory["favorite_symbol"]

    return DEFAULT_SYMBOL


def detect_interval(user_message, user_memory=None):
    text = user_message.lower()

    for key, interval in INTERVAL_MAP.items():
        if key in text:
            return interval

    if user_memory and user_memory.get("favorite_interval"):
        return user_memory["favorite_interval"]

    return DEFAULT_INTERVAL


def detect_user_intent(user_message):
    text = user_message.lower()

    if any(keyword.lower() in text for keyword in TRADING_PLAN_KEYWORDS):
        return "trade_plan"

    if any(keyword.lower() in text for keyword in MARKET_OVERVIEW_KEYWORDS):
        return "market_overview"

    if any(keyword.lower() in text for keyword in WHY_MOVE_KEYWORDS):
        return "why_move"

    if any(keyword.lower() in text for keyword in TEACHING_KEYWORDS):
        return "teaching"

    if any(keyword.lower() in text for keyword in RISK_MODE_KEYWORDS):
        return "risk_check"

    if any(word in text for word in ["今天", "明天", "非农", "初请", "cpi", "fomc", "美联储", "数据", "新闻"]):
        return "macro_news"

    return "general_market"


def is_market_overview_question(user_message):
    return detect_user_intent(user_message) == "market_overview"


def is_why_move_question(user_message):
    return detect_user_intent(user_message) == "why_move"


def is_teaching_question(user_message):
    return detect_user_intent(user_message) == "teaching"


def is_risk_check_question(user_message):
    return detect_user_intent(user_message) == "risk_check"


def is_multi_timeframe_question(user_message):
    text = user_message.lower()

    keywords = [
        "现在能买吗", "现在可以买", "现在能不能", "行情怎样", "怎么看",
        "适合进吗", "可以进吗", "可以买涨吗", "可以买跌吗",
        "做多", "做空", "回踩", "入场", "进场", "止损", "止盈",
        "目标", "反弹", "接多", "接空", "哪里买", "哪里卖",
        "什么价位", "追多", "追空", "突破", "跌破", "站稳",
        "数据", "新闻", "今晚", "今天", "明天", "有什么事", "重要事件",
        "非农", "初请", "cpi", "fomc", "利率",
        "怎么做", "交易计划", "计划", "策略", "做单", "布局"
    ]

    return any(word in text for word in keywords)


def get_user_memory(user_id):
    memory = load_json(MEMORY_FILE, {})

    if user_id not in memory:
        memory[user_id] = {
            "favorite_symbol": DEFAULT_SYMBOL,
            "favorite_interval": DEFAULT_INTERVAL,
            "risk_level": "normal",
            "message_count": 0,
            "last_question": ""
        }
        save_json(MEMORY_FILE, memory)

    return memory[user_id]


def update_user_memory(user_id, symbol, interval, user_message):
    memory = load_json(MEMORY_FILE, {})

    if user_id not in memory:
        memory[user_id] = {
            "favorite_symbol": symbol,
            "favorite_interval": interval,
            "risk_level": "normal",
            "message_count": 0,
            "last_question": ""
        }

    memory[user_id]["favorite_symbol"] = symbol
    memory[user_id]["favorite_interval"] = interval
    memory[user_id]["message_count"] += 1
    memory[user_id]["last_question"] = user_message

    if "保守" in user_message:
        memory[user_id]["risk_level"] = "conservative"
    elif "激进" in user_message:
        memory[user_id]["risk_level"] = "aggressive"
    elif "稳健" in user_message:
        memory[user_id]["risk_level"] = "normal"

    save_json(MEMORY_FILE, memory)
    return memory[user_id]


# =========================
# V12 Chinese News Layer
# =========================

def fetch_sina_7x24_news(limit=30):
    cache = load_json(CHINESE_NEWS_CACHE_FILE, {})
    now = time.time()

    if cache.get("created_at") and now - cache.get("created_at", 0) < 300:
        return cache.get("items", [])

    url = "https://finance.sina.com.cn/7x24/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Fix Chinese encoding issue.
        response.encoding = response.apparent_encoding or "gbk"
        html = response.text

        clean_text = strip_html(html)

        pattern = r"(\d{2}:\d{2}:\d{2})\s+(.{8,240}?)(?=\s+\d{2}:\d{2}:\d{2}\s+|$)"
        matches = re.findall(pattern, clean_text)

        items = []
        seen = set()

        for time_value, content in matches:
            content = content.strip()

            if len(content) < 8:
                continue

            key = f"{time_value}_{content[:50]}"
            if key in seen:
                continue

            seen.add(key)
            items.append({
                "time": time_value,
                "source": "新浪财经7x24",
                "content": content
            })

            if len(items) >= limit:
                break

        save_json(CHINESE_NEWS_CACHE_FILE, {"created_at": now, "items": items})
        return items

    except Exception as e:
        print("Chinese News Error:", e)
        return cache.get("items", [])


def is_chinese_high_impact_news(item):
    text = item.get("content", "").lower()
    return any(keyword.lower() in text for keyword in CHINESE_HIGH_IMPACT_KEYWORDS)


def is_chinese_news_relevant_to_symbol(item, symbol):
    text = item.get("content", "").lower()

    keywords = SYMBOL_NEWS_KEYWORDS.get(symbol, [])

    if any(keyword.lower() in text for keyword in keywords):
        return True

    if symbol.endswith("USDT"):
        return any(word in text for word in ["比特币", "btc", "加密", "crypto", "etf", "美联储", "美元"])

    if symbol in USD_SENSITIVE_SYMBOLS:
        return any(word in text for word in ["美元", "美联储", "cpi", "pce", "非农", "通胀", "美债"])

    return False


def get_chinese_news_risk(symbol):
    items = fetch_sina_7x24_news(limit=40)

    relevant = []
    high_impact = []

    for item in items:
        if is_chinese_high_impact_news(item):
            high_impact.append(item)

        if is_chinese_news_relevant_to_symbol(item, symbol):
            relevant.append(item)

    selected = relevant[:6] if relevant else high_impact[:6]

    if not selected:
        return {
            "has_risk": False,
            "summary": "中文快讯暂时没有检测到明显相关的高影响消息。",
            "items": []
        }

    lines = []
    for item in selected:
        lines.append(f"{item.get('time', '')}｜{item.get('content', '')}")

    return {
        "has_risk": True,
        "summary": "\n".join(lines),
        "items": selected
    }


def build_chinese_news_text(symbol):
    risk = get_chinese_news_risk(symbol)

    if not risk["has_risk"]:
        return risk["summary"]

    return f"""
中文财经快讯检测到相关消息：

{risk['summary']}

风控提示：
如果消息涉及美联储、CPI、非农、美元、美债、战争或 ETF，短线波动可能明显放大。
入场前最好降低仓位，等价格反应稳定后再判断。
"""


# =========================
# V12 Macro Calendar Engine
# =========================

def parse_macro_value(value):
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text.lower() in ["n/a", "na", "-"]:
        return None

    multiplier = 1.0
    clean = text.replace(",", "").replace("%", "").strip()

    if clean.endswith("K") or clean.endswith("k"):
        multiplier = 1000
        clean = clean[:-1]
    elif clean.endswith("M") or clean.endswith("m"):
        multiplier = 1000000
        clean = clean[:-1]
    elif clean.endswith("B") or clean.endswith("b"):
        multiplier = 1000000000
        clean = clean[:-1]

    try:
        return float(clean) * multiplier
    except Exception:
        return None


def normalize_macro_event(raw):
    title = normalize_text(raw.get("title") or raw.get("event") or raw.get("name") or raw.get("Event"))
    country = normalize_text(raw.get("country") or raw.get("Country"))
    date = normalize_text(raw.get("date") or raw.get("Date"))
    time_value = normalize_text(raw.get("time") or raw.get("Time"))
    impact = normalize_text(raw.get("impact") or raw.get("Impact") or raw.get("importance") or raw.get("Importance"))
    actual = normalize_text(raw.get("actual") or raw.get("Actual"))
    forecast = normalize_text(raw.get("forecast") or raw.get("Forecast") or raw.get("consensus") or raw.get("Consensus"))
    previous = normalize_text(raw.get("previous") or raw.get("Previous"))

    return {
        "title": title,
        "country": country,
        "date": date,
        "time": time_value,
        "impact": impact,
        "actual": actual,
        "forecast": forecast,
        "previous": previous,
        "source": normalize_text(raw.get("source") or "Macro Calendar")
    }


def fetch_forexfactory_calendar(days="today"):
    """
    Unofficial/free endpoint used by many calendar tools.
    If this endpoint changes, the function fails gracefully.
    """
    cache = load_json(MACRO_CACHE_FILE, {})
    cache_key = f"forexfactory_{days}_{datetime.utcnow().date().isoformat()}"
    now = time.time()

    if cache.get("key") == cache_key and now - cache.get("created_at", 0) < 900:
        return cache.get("events", [])

    urls = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
    ]

    for url in urls:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            data = response.json()

            events = []
            today = datetime.utcnow().date()
            target_dates = {today}

            if days == "tomorrow":
                target_dates = {today + timedelta(days=1)}
            elif days == "week":
                target_dates = {today + timedelta(days=i) for i in range(0, 7)}
            elif days == "today_tomorrow":
                target_dates = {today, today + timedelta(days=1)}

            for item in data:
                date_text = item.get("date", "")

                try:
                    event_dt = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
                    event_date = event_dt.date()
                    time_value = event_dt.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    event_date = today
                    time_value = date_text

                if event_date not in target_dates:
                    continue

                raw = {
                    "title": item.get("title"),
                    "country": item.get("country"),
                    "date": str(event_date),
                    "time": time_value,
                    "impact": item.get("impact"),
                    "actual": item.get("actual"),
                    "forecast": item.get("forecast"),
                    "previous": item.get("previous"),
                    "source": "ForexFactory"
                }

                events.append(normalize_macro_event(raw))

            save_json(MACRO_CACHE_FILE, {
                "key": cache_key,
                "created_at": now,
                "events": events
            })

            return events

        except Exception as e:
            print("ForexFactory Error:", e)

    return cache.get("events", [])


def is_macro_high_impact(event):
    text = f"{event.get('title', '')} {event.get('impact', '')}".lower()

    if any(keyword in text for keyword in HIGH_IMPACT_KEYWORDS):
        return True

    impact = str(event.get("impact", "")).lower()
    return "high" in impact or impact in ["3", "red"]


def is_macro_relevant_to_symbol(event, symbol):
    text = f"{event.get('title', '')} {event.get('country', '')}".lower()
    country = event.get("country", "").lower()

    if symbol in USD_SENSITIVE_SYMBOLS or symbol.endswith("USDT"):
        if country in ["usd", "us", "united states"] or "usd" in text or "united states" in text:
            return True

    if symbol == "EURUSD=X":
        return "eur" in country or "euro" in text or "ecb" in text

    if symbol == "GBPUSD=X":
        return "gbp" in country or "united kingdom" in text or "boe" in text

    if symbol == "JPY=X":
        return "jpy" in country or "japan" in text or "boj" in text

    return False


def filter_macro_events(kind=None, days="today_tomorrow", symbol=None):
    events = fetch_forexfactory_calendar(days=days)

    if kind:
        aliases = MACRO_EVENT_ALIASES.get(kind, [kind])
        events = [
            event for event in events
            if any(alias.lower() in event.get("title", "").lower() for alias in aliases)
        ]

    if symbol:
        relevant = [event for event in events if is_macro_relevant_to_symbol(event, symbol)]
        high = [event for event in events if is_macro_high_impact(event)]
        events = relevant if relevant else high

    return events


def explain_macro_event(event):
    title = event.get("title", "")
    actual = event.get("actual", "")
    forecast = event.get("forecast", "")
    previous = event.get("previous", "")

    actual_num = parse_macro_value(actual)
    forecast_num = parse_macro_value(forecast)

    lower_title = title.lower()

    if not actual:
        return "实际值还没公布，数据前后波动可能会放大。"

    if actual_num is None or forecast_num is None:
        return "实际值已公布，但暂时无法和预测做数值比较。"

    stronger_than_expected = actual_num > forecast_num
    weaker_than_expected = actual_num < forecast_num

    if any(word in lower_title for word in ["jobless", "unemployment", "失业"]):
        if stronger_than_expected:
            return "实际高于预期，通常代表就业压力增加，偏利空美元，黄金/BTC 可能获得支撑。"
        if weaker_than_expected:
            return "实际低于预期，通常代表就业仍强，偏利多美元，黄金/BTC 可能承压。"

    if any(word in lower_title for word in ["cpi", "pce", "ppi", "inflation", "price", "通胀"]):
        if stronger_than_expected:
            return "实际高于预期，通胀压力偏强，市场可能降低降息预期，美元偏强，黄金/BTC 可能承压。"
        if weaker_than_expected:
            return "实际低于预期，通胀压力缓和，降息预期可能升温，黄金/BTC 可能获得支撑。"

    if any(word in lower_title for word in ["payroll", "employment", "non-farm", "非农"]):
        if stronger_than_expected:
            return "实际高于预期，就业强劲，偏利多美元，黄金/BTC 可能承压。"
        if weaker_than_expected:
            return "实际低于预期，就业走弱，偏利空美元，黄金/BTC 可能获得支撑。"

    if stronger_than_expected:
        return "实际高于预期，通常会带来短线波动，需要看美元和美债反应。"

    if weaker_than_expected:
        return "实际低于预期，通常会带来短线波动，需要看美元和美债反应。"

    return "实际接近预期，市场可能更关注细节和后续讲话。"


def translate_macro_title(title):
    title = normalize_text(title)

    if title in MACRO_TRANSLATION:
        return MACRO_TRANSLATION[title]

    lower_title = title.lower()

    if "non-farm" in lower_title or "nonfarm" in lower_title or "payroll" in lower_title:
        return "美国非农就业人数"

    if "initial jobless" in lower_title or "jobless claims" in lower_title:
        return "美国初请失业金人数"

    if "continuing jobless" in lower_title:
        return "美国续请失业金人数"

    if "core cpi" in lower_title:
        return "美国核心 CPI 通胀数据"

    if "cpi" in lower_title or "consumer price" in lower_title:
        return "美国 CPI 通胀数据"

    if "core pce" in lower_title:
        return "美国核心 PCE 物价指数"

    if "pce" in lower_title:
        return "美国 PCE 物价指数"

    if "ppi" in lower_title or "producer price" in lower_title:
        return "美国 PPI 生产者物价指数"

    if "federal funds" in lower_title or "rate decision" in lower_title:
        return "美联储利率决议"

    if "fomc" in lower_title:
        return "FOMC 美联储会议"

    if "powell" in lower_title:
        return "美联储主席鲍威尔讲话"

    if "retail sales" in lower_title:
        return "美国零售销售"

    if "gdp" in lower_title:
        return "美国 GDP 数据"

    if "pmi" in lower_title:
        return "PMI 采购经理人指数"

    return title


def translate_country(country):
    country = normalize_text(country)
    return COUNTRY_TRANSLATION.get(country, country)


def translate_impact(impact):
    impact = normalize_text(impact)
    lower = impact.lower()

    if "high" in lower or lower in ["3", "red"]:
        return "高影响"

    if "medium" in lower or lower in ["2", "orange"]:
        return "中影响"

    if "low" in lower or lower in ["1", "yellow"]:
        return "低影响"

    return impact or "未标注"


def format_macro_event(event):
    country = translate_country(event.get("country", ""))
    title = translate_macro_title(event.get("title", ""))
    impact = translate_impact(event.get("impact", ""))

    return f"""
{country}｜{title}

时间：{event.get('time', '')}
影响级别：{impact}
前值：{event.get('previous', '') or '暂无'}
市场预测：{event.get('forecast', '') or '暂无'}
实际值：{event.get('actual', '') or '等待公布'}
市场解读：{explain_macro_event(event)}
""".strip()


def build_macro_report(kind=None, days="today_tomorrow", symbol=None):
    events = filter_macro_events(kind=kind, days=days, symbol=symbol)

    if not events:
        return "暂时没有找到相关经济数据。"

    important = [event for event in events if is_macro_high_impact(event)]
    selected = important if important else events

    blocks = [format_macro_event(event) for event in selected[:8]]

    return "\n\n".join(blocks)


def get_macro_risk(symbol):
    events = filter_macro_events(days="today_tomorrow", symbol=symbol)
    high_events = [event for event in events if is_macro_high_impact(event)]

    selected = high_events[:6] if high_events else events[:4]

    if not selected:
        return {
            "has_risk": False,
            "summary": "未来 24~48 小时暂时没有检测到明显高影响经济数据。",
            "events": []
        }

    lines = []
    for event in selected:
        lines.append(
            f"{event.get('time', '')}｜{translate_country(event.get('country', ''))}｜{translate_macro_title(event.get('title', ''))}｜"
            f"前值:{event.get('previous', '') or '暂无'}｜市场预测:{event.get('forecast', '') or '暂无'}｜实际值:{event.get('actual', '') or '待公布'}"
        )

    return {
        "has_risk": True,
        "summary": "\n".join(lines),
        "events": selected
    }


def build_news_risk_text(symbol):
    chinese_news_text = build_chinese_news_text(symbol)
    macro_risk = get_macro_risk(symbol)

    if not macro_risk["has_risk"]:
        macro_text = macro_risk["summary"]
    else:
        macro_text = f"""
未来 24~48 小时检测到可能影响行情的重要经济数据：

{macro_risk['summary']}

宏观风控：
数据公布前后 5~15 分钟波动可能放大，不建议提前重仓进场。
如果实际值和预测差距较大，黄金、美元、BTC、外汇都可能快速波动。
"""

    return f"""
【中文快讯】
{chinese_news_text}

【宏观数据】
{macro_text}
"""


# =========================
# Market Data + Technical Layer
# =========================

def get_klines(symbol, interval):
    if is_crypto_symbol(symbol):
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": LIMIT}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])

        df = df[["open", "high", "low", "close", "volume"]].copy()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df.tail(LIMIT)

    yf_interval_map = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
    yf_period_map = {"15m": "5d", "1h": "1mo", "4h": "1mo", "1d": "6mo"}

    yf_interval = yf_interval_map.get(interval, "1h")
    yf_period = yf_period_map.get(interval, "1mo")

    df = yf.download(
        symbol,
        period=yf_period,
        interval=yf_interval,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        raise Exception(f"No data found for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(col).lower() for col in df.columns]

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise Exception(f"Missing columns: {missing}")

    df = df[required_cols].copy()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    if interval == "4h":
        df.index = pd.to_datetime(df.index)
        df = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

    return df.tail(LIMIT)


def calc_fibonacci(high_value, low_value):
    diff = high_value - low_value

    return {
        "fib_236": high_value - diff * 0.236,
        "fib_382": high_value - diff * 0.382,
        "fib_500": high_value - diff * 0.5,
        "fib_618": high_value - diff * 0.618,
        "fib_786": high_value - diff * 0.786,
    }


def detect_price_action(close, high, low):
    last_close = float(close.iloc[-1])
    prev_high = float(high.tail(20).iloc[:-1].max())
    prev_low = float(low.tail(20).iloc[:-1].min())

    recent_high = float(high.tail(50).max())
    recent_low = float(low.tail(50).min())

    if last_close > prev_high:
        structure_event = "BOS 向上，短线结构有转强迹象"
    elif last_close < prev_low:
        structure_event = "BOS 向下，短线结构有转弱迹象"
    elif high.iloc[-1] > prev_high and last_close < prev_high:
        structure_event = "上方有扫高后回落，可能是假突破/扫流动性"
    elif low.iloc[-1] < prev_low and last_close > prev_low:
        structure_event = "下方有扫低后收回，可能是假跌破/扫流动性"
    else:
        structure_event = "结构暂时没有明显突破"

    if abs(high.tail(10).max() - prev_high) / max(last_close, 0.0001) < 0.002:
        liquidity = "上方有等高流动性"
    elif abs(low.tail(10).min() - prev_low) / max(last_close, 0.0001) < 0.002:
        liquidity = "下方有等低流动性"
    else:
        liquidity = "流动性位置不算特别明显"

    mid = (recent_high + recent_low) / 2

    if last_close > mid:
        premium_discount = "价格处在区间偏高位置，追多性价比下降"
    else:
        premium_discount = "价格处在区间偏低位置，追空性价比下降"

    return structure_event, liquidity, premium_discount


def analyze_market(symbol, interval):
    df = get_klines(symbol, interval)

    close = pd.Series(df["close"].astype(float).to_numpy().flatten())
    high = pd.Series(df["high"].astype(float).to_numpy().flatten())
    low = pd.Series(df["low"].astype(float).to_numpy().flatten())

    current_price = float(close.iloc[-1])

    ma20 = SMAIndicator(close=close, window=20).sma_indicator().iloc[-1]
    ma50 = SMAIndicator(close=close, window=50).sma_indicator().iloc[-1]
    ema20 = EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]

    rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

    macd_obj = MACD(close=close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_signal = macd_obj.macd_signal().iloc[-1]

    atr_obj = AverageTrueRange(high=high, low=low, close=close, window=14)
    atr = atr_obj.average_true_range().iloc[-1]

    support = float(low.tail(30).min())
    resistance = float(high.tail(30).max())

    swing_low = float(low.tail(50).min())
    swing_high = float(high.tail(50).max())

    fib = calc_fibonacci(swing_high, swing_low)

    structure_event, liquidity, premium_discount = detect_price_action(close, high, low)

    bull_score = 0
    bear_score = 0

    if current_price > ma20:
        bull_score += 1
    else:
        bear_score += 1

    if current_price > ma50:
        bull_score += 1
    else:
        bear_score += 1

    if current_price > ema20:
        bull_score += 1
    else:
        bear_score += 1

    if macd_line > macd_signal:
        bull_score += 1
    else:
        bear_score += 1

    if rsi > 55:
        bull_score += 1
    elif rsi < 45:
        bear_score += 1

    if current_price > fib["fib_500"]:
        bull_score += 1
    else:
        bear_score += 1

    if "向上" in structure_event or "扫低" in structure_event:
        bull_score += 1
    elif "向下" in structure_event or "扫高" in structure_event:
        bear_score += 1

    if bull_score >= 5:
        trend = "偏多"
    elif bear_score >= 5:
        trend = "偏空"
    else:
        trend = "震荡"

    total = bull_score + bear_score
    long_probability = 50 if total == 0 else int((bull_score / total) * 100)
    short_probability = 100 - long_probability

    if rsi > 75:
        risk = "超买风险偏高，不建议追多"
    elif rsi < 25:
        risk = "超卖风险偏高，不建议追空"
    elif abs(current_price - resistance) / current_price < 0.003:
        risk = "价格贴近压力位，追多风险偏高"
    elif abs(current_price - support) / current_price < 0.003:
        risk = "价格贴近支撑位，追空风险偏高"
    else:
        risk = "风险中等"

    if trend == "偏多":
        long_zone_low = min(fib["fib_618"], support + atr * 0.2, ema50)
        long_zone_high = max(fib["fib_500"], ema20, support + atr * 0.8)
    else:
        long_zone_low = min(support + atr * 0.2, fib["fib_618"], ema50)
        long_zone_high = max(support + atr * 0.8, fib["fib_500"], ema20)

    if long_zone_low > long_zone_high:
        long_zone_low, long_zone_high = long_zone_high, long_zone_low

    short_zone_low = min(resistance - atr * 0.8, fib["fib_500"], ema20)
    short_zone_high = max(resistance - atr * 0.2, fib["fib_382"], ema50)

    if short_zone_low > short_zone_high:
        short_zone_low, short_zone_high = short_zone_high, short_zone_low

    stop_loss_long = min(support - atr * 0.5, long_zone_low - atr * 0.8)
    stop_loss_short = max(resistance + atr * 0.5, short_zone_high + atr * 0.8)

    target_1_long = resistance
    target_2_long = resistance + atr * 1.5
    target_1_short = support
    target_2_short = support - atr * 1.5

    rr_long = (target_1_long - long_zone_high) / max(long_zone_high - stop_loss_long, 0.0001)
    rr_short = (short_zone_low - target_1_short) / max(stop_loss_short - short_zone_low, 0.0001)

    long_rr_comment = "做多风险收益比不错" if rr_long >= 2 else ("做多风险收益比一般" if rr_long >= 1 else "做多风险收益比偏差")
    short_rr_comment = "做空风险收益比不错" if rr_short >= 2 else ("做空风险收益比一般" if rr_short >= 1 else "做空风险收益比偏差")

    if trend == "偏多":
        entry_bias = "优先等回踩做多"
    elif trend == "偏空":
        entry_bias = "优先等反弹做空"
    else:
        entry_bias = "震荡盘优先观望，等区间突破"

    long_confirm = "回踩入场区后，15m 收回 EMA20 上方，或出现扫低收回/止跌阳线，再考虑轻仓。"
    short_confirm = "反弹入场区后，15m 无法站稳压力，或出现扫高回落/受压阴线，再考虑轻仓。"

    long_invalid = f"如果跌破 {round(stop_loss_long, 4)}，多单思路先失效。"
    short_invalid = f"如果突破 {round(stop_loss_short, 4)}，空单思路先失效。"

    return {
        "symbol": symbol,
        "interval": interval,
        "price": round(current_price, 4),
        "ma20": round(float(ma20), 4),
        "ma50": round(float(ma50), 4),
        "ema20": round(float(ema20), 4),
        "ema50": round(float(ema50), 4),
        "rsi": round(float(rsi), 2),
        "atr": round(float(atr), 4),
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "swing_low": round(swing_low, 4),
        "swing_high": round(swing_high, 4),
        "fib_236": round(fib["fib_236"], 4),
        "fib_382": round(fib["fib_382"], 4),
        "fib_500": round(fib["fib_500"], 4),
        "fib_618": round(fib["fib_618"], 4),
        "fib_786": round(fib["fib_786"], 4),
        "trend": trend,
        "risk": risk,
        "structure_event": structure_event,
        "liquidity": liquidity,
        "premium_discount": premium_discount,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "long_probability": long_probability,
        "short_probability": short_probability,
        "entry_long_low": round(long_zone_low, 4),
        "entry_long_high": round(long_zone_high, 4),
        "entry_short_low": round(short_zone_low, 4),
        "entry_short_high": round(short_zone_high, 4),
        "stop_loss_long": round(stop_loss_long, 4),
        "stop_loss_short": round(stop_loss_short, 4),
        "target_1_long": round(target_1_long, 4),
        "target_2_long": round(target_2_long, 4),
        "target_1_short": round(target_1_short, 4),
        "target_2_short": round(target_2_short, 4),
        "rr_long": round(rr_long, 2),
        "rr_short": round(rr_short, 2),
        "long_rr_comment": long_rr_comment,
        "short_rr_comment": short_rr_comment,
        "entry_bias": entry_bias,
        "long_confirm": long_confirm,
        "short_confirm": short_confirm,
        "long_invalid": long_invalid,
        "short_invalid": short_invalid,
    }


def analyze_multi_timeframe(symbol):
    results = {}

    for interval in MULTI_TIMEFRAMES:
        try:
            results[interval] = analyze_market(symbol, interval)
        except Exception as e:
            print(f"Error {symbol} {interval}:", e)

    return results


def build_multi_timeframe_summary(results):
    if not results:
        return {
            "overall": "数据不足",
            "avg_long": 50,
            "avg_short": 50,
            "trend_text": "暂无有效周期数据"
        }

    long_total = 0
    short_total = 0
    trend_texts = []

    for interval, data in results.items():
        long_total += data["long_probability"]
        short_total += data["short_probability"]
        trend_texts.append(f"{interval}：{data['trend']}")

    avg_long = int(long_total / len(results))
    avg_short = int(short_total / len(results))

    if avg_long >= 65:
        overall = "整体偏多"
    elif avg_short >= 65:
        overall = "整体偏空"
    else:
        overall = "整体偏震荡"

    return {
        "overall": overall,
        "avg_long": avg_long,
        "avg_short": avg_short,
        "trend_text": " / ".join(trend_texts)
    }


def ensure_multi_tf_data(symbol, interval, multi_tf_data):
    if not multi_tf_data:
        single_data = analyze_market(symbol, interval)
        return {"15m": single_data, "1h": single_data, "4h": single_data}

    reference = next(iter(multi_tf_data.values()))

    for tf in MULTI_TIMEFRAMES:
        if tf not in multi_tf_data:
            multi_tf_data[tf] = reference

    return multi_tf_data



def is_trading_plan_question(user_message):
    text = user_message.lower()
    return any(keyword.lower() in text for keyword in TRADING_PLAN_KEYWORDS)


def build_trade_plan_text(symbol, data, summary, news_risk_text):
    trend = data["trend"]
    price = data["price"]

    long_low = data["entry_long_low"]
    long_high = data["entry_long_high"]
    long_sl = data["stop_loss_long"]
    long_t1 = data["target_1_long"]
    long_t2 = data["target_2_long"]
    long_rr = data["rr_long"]

    short_low = data["entry_short_low"]
    short_high = data["entry_short_high"]
    short_sl = data["stop_loss_short"]
    short_t1 = data["target_1_short"]
    short_t2 = data["target_2_short"]
    short_rr = data["rr_short"]

    if trend == "偏多":
        main_plan = f"""
计划A：回踩做多
入场区：{long_low} ~ {long_high}
止损：{long_sl}
目标1：{long_t1}
目标2：{long_t2}
风险收益比：{long_rr}

触发条件：
回踩后 15m 收回 EMA20，或出现扫低收回/止跌阳线，再考虑轻仓。
"""
        backup_plan = f"""
计划B：多头失效后等待反抽
如果跌破 {long_sl}，多单思路先放弃。
后面等反抽失败，再看 {short_low} ~ {short_high} 附近是否有做空机会。
空单止损参考：{short_sl}
空单目标：{short_t1} / {short_t2}
"""
    elif trend == "偏空":
        main_plan = f"""
计划A：反弹做空
入场区：{short_low} ~ {short_high}
止损：{short_sl}
目标1：{short_t1}
目标2：{short_t2}
风险收益比：{short_rr}

触发条件：
反弹后 15m 无法站稳压力，或出现扫高回落/受压阴线，再考虑轻仓。
"""
        backup_plan = f"""
计划B：空头失效后等待回踩
如果突破 {short_sl}，空单思路先放弃。
后面等回踩不破，再看 {long_low} ~ {long_high} 附近是否有做多机会。
多单止损参考：{long_sl}
多单目标：{long_t1} / {long_t2}
"""
    else:
        main_plan = f"""
计划A：区间低吸高抛
做多观察区：{long_low} ~ {long_high}
多单止损：{long_sl}
多单目标：{long_t1}

做空观察区：{short_low} ~ {short_high}
空单止损：{short_sl}
空单目标：{short_t1}

触发条件：
震荡盘不要追，必须等靠近区间边缘后有确认再做。
"""
        backup_plan = f"""
计划B：等突破后顺势
如果站稳 {data["resistance"]} 上方，再看回踩确认做多。
如果跌破 {data["support"]} 下方，再看反抽确认做空。
中间位置不建议硬追，容易两边扫。
"""

    return f"""
当前价格：{price}
整体方向：{summary['overall']}
多周期结构：{summary['trend_text']}

新闻/数据风控：
{news_risk_text}

{main_plan}

{backup_plan}

仓位建议：
如果数据/新闻风险较大，只能轻仓试，不适合重仓。
如果入场后没有按预期走，先按止损纪律处理，不要扛单。
"""


def safe_analyze_symbol(symbol):
    try:
        return analyze_market(symbol, "15m")
    except Exception as e:
        print(f"Overview Error {symbol}:", e)
        return None


def build_market_overview_data():
    symbols = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "黄金": "GC=F",
        "白银": "SI=F",
        "EURUSD": "EURUSD=X",
        "USDJPY": "JPY=X",
    }

    rows = []

    for name, symbol in symbols.items():
        data = safe_analyze_symbol(symbol)

        if not data:
            continue

        rows.append({
            "name": name,
            "symbol": symbol,
            "price": data["price"],
            "trend": data["trend"],
            "rsi": data["rsi"],
            "support": data["support"],
            "resistance": data["resistance"],
            "structure": data["structure_event"],
            "risk": data["risk"],
        })

    return rows


def format_market_overview_rows(rows):
    if not rows:
        return "暂时无法读取市场总览数据。"

    lines = []

    for row in rows:
        lines.append(
            f"{row['name']}：价格 {row['price']}，趋势 {row['trend']}，RSI {row['rsi']}，"
            f"支撑 {row['support']}，压力 {row['resistance']}，结构：{row['structure']}"
        )

    return "\n".join(lines)


def generate_market_overview_reply(user_message, user_memory):
    rows = build_market_overview_data()
    overview_text = format_market_overview_rows(rows)

    symbol = user_memory.get("favorite_symbol", DEFAULT_SYMBOL)
    news_risk_text = build_news_risk_text(symbol)

    prompt = f"""
{SYSTEM_PROMPT}

用户问：
{user_message}

用户风险偏好：
{user_memory.get("risk_level")}

市场总览数据：
{overview_text}

新闻/宏观风险：
{news_risk_text}

请像真人交易员一样，给一个自然的市场总览。

要求：
- 不要列太多表格
- 用 120~200 字
- 重点说现在是 Risk On / Risk Off / 震荡观望
- 说清楚今天适不适合交易
- 如果新闻数据风险大，先提醒
- 给出 1~2 个更值得关注的品种
- 不要喊单
- 最后写：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.45
    )

    return response.choices[0].message.content


def generate_flexible_market_reply(user_message, symbol, multi_tf_data, summary, user_memory, news_risk_text):
    d15 = multi_tf_data["15m"]
    intent = detect_user_intent(user_message)
    asset_name = get_asset_name(symbol)
    asset_macro_note = get_asset_macro_note(symbol)

    prompt = f"""
{SYSTEM_PROMPT}

你现在是一个真人交易员型 AI，不要每次都用固定模板。
请根据用户意图，用自然聊天方式回复。

用户问题：
{user_message}

识别到的意图：
{intent}

品种：
{asset_name}

品种风控重点：
{asset_macro_note}

用户信息：
风险偏好：{user_memory.get("risk_level")}
常看品种：{user_memory.get("favorite_symbol")}

新闻/宏观风控：
{news_risk_text}

行情数据：
当前价格：{d15['price']}
15m 趋势：{d15['trend']}
整体方向：{summary['overall']}
做多概率：{summary['avg_long']}%
做空概率：{summary['avg_short']}%
多周期结构：{summary['trend_text']}
RSI：{d15['rsi']}
支撑：{d15['support']}
压力：{d15['resistance']}
结构事件：{d15['structure_event']}
流动性：{d15['liquidity']}
高低估：{d15['premium_discount']}
风险：{d15['risk']}

交易参考：
回踩做多区：{d15['entry_long_low']} ~ {d15['entry_long_high']}
多单止损：{d15['stop_loss_long']}
多单目标：{d15['target_1_long']} / {d15['target_2_long']}
反弹做空区：{d15['entry_short_low']} ~ {d15['entry_short_high']}
空单止损：{d15['stop_loss_short']}
空单目标：{d15['target_1_short']} / {d15['target_2_short']}

回复规则：
- 如果意图是 why_move：重点解释为什么涨/跌，结合新闻、美元、美债、ETF、结构和情绪，不要硬给入场计划。
- 如果意图是 risk_check：重点说能不能追、风险在哪里、等什么确认。
- 如果意图是 teaching：用简单话解释概念，再结合当前行情举例。
- 如果意图是 macro_news：重点解释数据/新闻影响。
- 如果用户没有要求计划A/B，不要强行输出计划A/B。
- 语气像真人交易员，简短、有判断。
- 控制在 120~220 字。
- 不要说保证、一定、稳赢。
- 最后写：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content


def generate_trade_plan_reply(user_message, symbol, multi_tf_data, summary, user_memory, news_risk_text):
    d15 = multi_tf_data["15m"]
    plan_text = build_trade_plan_text(symbol, d15, summary, news_risk_text)
    asset_name = get_asset_name(symbol)
    asset_macro_note = get_asset_macro_note(symbol)

    prompt = f"""
{SYSTEM_PROMPT}

用户问：
{user_message}

品种：
{asset_name}

品种风控重点：
{asset_macro_note}

请根据下面数据，输出一个标准化交易计划。

交易计划数据：
{plan_text}

你必须严格使用下面格式，不要改格式，不要写成长段落：

【{asset_name} 交易计划】

当前看法：
用 2~3 句话说明当前方向、是否适合追单、有没有新闻数据风险。

计划A：主计划
方向：做多/做空/观望
入场区：
止损：
目标1：
目标2：
触发条件：
失效条件：

计划B：备用计划
方向：做多/做空/观望
入场区：
止损：
目标1：
目标2：
触发条件：
失效条件：

风控：
1. 如果是黄金/白银，必须提醒美元、美债、CPI、非农、美联储风险。
2. 如果是外汇，必须提醒美元和央行数据风险。
3. 如果是 BTC/ETH，加上美元、美债、ETF 和风险情绪提醒。
4. 如果位置不好，明确说不要追。
5. 仓位建议必须保守，不可以叫重仓。

要求：
- 必须分行
- 必须有计划A和计划B
- 必须有入场区、止损、目标、触发条件、失效条件
- 不要说保证、一定、稳赢
- 最后必须写：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.25
    )

    return response.choices[0].message.content


def generate_ai_reply(user_message, symbol, multi_tf_data, summary, user_memory, news_risk_text):
    d15 = multi_tf_data["15m"]
    d1h = multi_tf_data["1h"]
    d4h = multi_tf_data["4h"]

    prompt = f"""
{SYSTEM_PROMPT}

用户问题：
{user_message}

用户信息：
风险偏好：{user_memory.get("risk_level")}
常看品种：{user_memory.get("favorite_symbol")}

V12 宏观/新闻风控：
{news_risk_text}

多周期行情：
15m：趋势 {d15['trend']}，价格 {d15['price']}，RSI {d15['rsi']}，ATR {d15['atr']}
1h：趋势 {d1h['trend']}，价格 {d1h['price']}，RSI {d1h['rsi']}
4h：趋势 {d4h['trend']}，价格 {d4h['price']}，RSI {d4h['rsi']}

综合结果：
整体方向：{summary['overall']}
做多概率：{summary['avg_long']}%
做空概率：{summary['avg_short']}
多周期结构：{summary['trend_text']}

Price Action / SMC：
结构事件：{d15['structure_event']}
流动性：{d15['liquidity']}
高低估：{d15['premium_discount']}

当前优先思路：
{d15['entry_bias']}

回踩做多区域：
{d15['entry_long_low']} ~ {d15['entry_long_high']}
多单止损：{d15['stop_loss_long']}
多单目标：{d15['target_1_long']} / {d15['target_2_long']}
做多RR：{d15['rr_long']}，{d15['long_rr_comment']}
多单确认：{d15['long_confirm']}
多单失效：{d15['long_invalid']}

反弹做空区域：
{d15['entry_short_low']} ~ {d15['entry_short_high']}
空单止损：{d15['stop_loss_short']}
空单目标：{d15['target_1_short']} / {d15['target_2_short']}
做空RR：{d15['rr_short']}，{d15['short_rr_comment']}
空单确认：{d15['short_confirm']}
空单失效：{d15['short_invalid']}

请像真人交易员一样回复。

要求：
- 控制在 150~230 字
- 如果有宏观数据或新闻风险，必须先提醒
- 如果用户问初请、非农、CPI、FOMC，要重点说明 实际值/市场预测/前值 和影响
- 如果用户问入场，要结合数据风险判断是否适合提前进
- 不要像 AI 报告
- 不要机械列指标
- 不要说“保证”“一定”“稳赢”
- 给参考，不喊单

最后自然加一句：
“以上仅供行情参考，不构成投资建议。”
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content


def build_chart_prompt(caption, user_memory, news_risk_text):
    return f"""
{VISION_PROMPT}

用户附带文字：
{caption}

用户信息：
风险偏好：{user_memory.get("risk_level")}
常看品种：{user_memory.get("favorite_symbol")}

宏观/新闻风控：
{news_risk_text}

请根据图片进行专业图表分析。

特别注意：
- 如果看到多窗口图，要分别判断主要窗口，再给综合结论
- 如果是黄金/外汇/加密图，重点看结构，不要乱猜精确价位
- 如果价格数字模糊，不要硬报具体点位
- 如果有重要数据，必须提醒数据公布前后波动会放大
- 优先说：现在能不能追、等哪里确认、失效条件是什么
- 可提到 BOS / CHOCH / OB / FVG / liquidity sweep，但不要强行编造
- 控制在 120~220 字
"""


def image_to_data_url(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def generate_vision_reply(prompt, image_data_url):
    response = client.chat.completions.create(
        model=VISION_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def subscribe_alert(user_id, chat_id, symbol):
    alerts = load_json(ALERT_FILE, [])

    for item in alerts:
        if item["user_id"] == user_id and item["symbol"] == symbol:
            return False

    alerts.append({
        "user_id": user_id,
        "chat_id": chat_id,
        "symbol": symbol,
        "last_alert_time": 0,
        "last_macro_alert_time": 0
    })

    save_json(ALERT_FILE, alerts)
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
你好，我是 AI 行情助手。

你可以这样问：

BTC 现在能买吗？
BTC 怎么做？
黄金怎么做？
EURUSD 怎么做？
ETH 回踩哪里做多？
黄金反弹哪里可以空？
今天有什么重要数据？
明天有什么数据？
初请失业金怎样？
明天非农怎么看？
CPI 会影响黄金吗？

V14 新增：
Full Macro Engine
- ForexFactory 经济日历抓取
- 实际值 / 市场预测 / 前值
- /today /tomorrow /nfp /cpi /jobless /fomc
- 中文快讯 + 宏观数据 + 技术面融合
- 突发新闻主动推送
- AI 自动交易计划A/B

也可以：
订阅 BTC 提醒
我的设置
/help
"""
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
支持功能：

/today 今日重要数据
/tomorrow 明日重要数据
/nfp 非农
/jobless 初请失业金
/cpi CPI
/fomc 美联储/FOMC

其他：
BTC 现在能买吗？
BTC 怎么做？
黄金怎么做？
EURUSD 怎么做？
黄金回踩哪里做多？
今天有什么重要数据？
明天非农怎么看？
这个图怎么看？直接发截图
订阅 BTC 提醒
我的设置
"""
    await update.message.reply_text(text)


async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    memory = get_user_memory(user_id)

    text = f"""
你的设置：

常看品种：
{memory.get('favorite_symbol')}

常看周期：
{memory.get('favorite_interval')}

风险偏好：
{memory.get('risk_level')}

历史提问次数：
{memory.get('message_count')}
"""
    await update.message.reply_text(text)


async def macro_command(update: Update, context: ContextTypes.DEFAULT_TYPE, kind=None, days="today_tomorrow"):
    user_id = str(update.message.from_user.id)
    memory = get_user_memory(user_id)
    symbol = memory.get("favorite_symbol", DEFAULT_SYMBOL)

    report = build_macro_report(kind=kind, days=days, symbol=symbol)

    await update.message.reply_text(
        f"{report}\n\n数据前后波动可能放大，不建议提前重仓。\n以上仅供行情参考，不构成投资建议。"
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind=None, days="today")


async def tomorrow_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind=None, days="tomorrow")


async def nfp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="nfp", days="week")


async def cpi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="cpi", days="week")


async def jobless_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="jobless", days="week")


async def fomc_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="fomc", days="week")



def get_breaking_news_items():
    items = fetch_sina_7x24_news(limit=50)
    selected = []

    for item in items:
        content = item.get("content", "").lower()

        if any(keyword.lower() in content for keyword in BREAKING_NEWS_KEYWORDS):
            selected.append(item)

    return selected[:10]


def classify_breaking_news_impact(content):
    text = content.lower()

    if any(word in text for word in ["cpi", "通胀", "高于预期", "低于预期", "非农", "初请", "利率决议", "fomc", "鲍威尔", "美联储"]):
        return "宏观数据/美联储风险，可能影响美元、黄金、BTC 和外汇。"

    if any(word in text for word in ["战争", "袭击", "爆炸", "地缘", "制裁", "避险"]):
        return "地缘风险升温，黄金可能出现避险波动，风险资产可能承压。"

    if any(word in text for word in ["etf", "比特币", "btc", "以太坊", "加密"]):
        return "加密市场相关新闻，BTC/ETH 可能短线放大波动。"

    if any(word in text for word in ["美元", "美债", "收益率"]):
        return "美元/美债相关消息，可能影响黄金、外汇和 BTC。"

    return "市场可能短线放大波动，先观察价格反应。"


def build_breaking_news_message(item):
    content = item.get("content", "")
    time_value = item.get("time", "")

    impact = classify_breaking_news_impact(content)

    return f"""
【突发快讯】
时间：{time_value}

{content}

可能影响：
{impact}

风控提醒：
消息刚出来时容易快速拉升或跳水，不建议第一时间重仓追。
先等 5~15 分钟，看价格是否站稳关键位。

以上仅供行情参考，不构成投资建议。
""".strip()


async def check_breaking_news(context: ContextTypes.DEFAULT_TYPE):
    alerts = load_json(ALERT_FILE, [])

    if not alerts:
        return

    state = load_json(BREAKING_NEWS_STATE_FILE, {
        "sent_keys": [],
        "last_check": 0
    })

    sent_keys = set(state.get("sent_keys", []))
    items = get_breaking_news_items()

    new_sent = []

    for item in items:
        content = item.get("content", "")
        time_value = item.get("time", "")
        key = f"{time_value}_{content[:80]}"

        if key in sent_keys:
            continue

        message = build_breaking_news_message(item)

        sent_chat_ids = set()

        for sub in alerts:
            chat_id = sub.get("chat_id")

            if not chat_id or chat_id in sent_chat_ids:
                continue

            try:
                await context.bot.send_message(chat_id=chat_id, text=message)
                sent_chat_ids.add(chat_id)
            except Exception as e:
                print("Breaking News Send Error:", e)

        new_sent.append(key)

    all_keys = list(sent_keys) + new_sent
    all_keys = all_keys[-200:]

    save_json(BREAKING_NEWS_STATE_FILE, {
        "sent_keys": all_keys,
        "last_check": int(time.time())
    })


async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    alerts = load_json(ALERT_FILE, [])
    changed = False
    now = int(time.time())

    for item in alerts:
        try:
            last_alert_time = item.get("last_alert_time", 0)
            last_macro_alert_time = item.get("last_macro_alert_time", 0)

            symbol = item["symbol"]

            if now - last_macro_alert_time >= MACRO_ALERT_COOLDOWN_SECONDS:
                macro_risk = get_macro_risk(symbol)

                if macro_risk["has_risk"]:
                    message = f"""
重要数据提醒：{symbol}

未来 24~48 小时有可能影响行情的宏观数据：

{macro_risk['summary']}

数据前后波动容易放大，不建议提前重仓进场。
以上仅供行情参考，不构成投资建议。
"""
                    await context.bot.send_message(chat_id=item["chat_id"], text=message)
                    item["last_macro_alert_time"] = now
                    changed = True

            if now - last_alert_time < ALERT_COOLDOWN_SECONDS:
                continue

            results = analyze_multi_timeframe(symbol)
            summary = build_multi_timeframe_summary(results)

            should_alert = False
            reason = ""

            if summary["avg_long"] >= 75:
                should_alert = True
                reason = "多头开始明显增强。"
            elif summary["avg_short"] >= 75:
                should_alert = True
                reason = "空头开始明显增强。"

            if should_alert:
                message = f"""
行情提醒：{symbol}

整体方向：{summary['overall']}
做多概率：{summary['avg_long']}%
做空概率：{summary['avg_short']}%

{reason}

别急着追，先等关键位确认。
以上仅供行情参考，不构成投资建议。
"""
                await context.bot.send_message(chat_id=item["chat_id"], text=message)
                item["last_alert_time"] = now
                changed = True

        except Exception as e:
            print("Alert Error:", e)

    if changed:
        save_json(ALERT_FILE, alerts)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    current_memory = get_user_memory(user_id)
    caption = update.message.caption or ""

    symbol = detect_symbol(caption, current_memory)

    try:
        await update.message.reply_text("收到图了，我先帮你看一下结构...")

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name

        await file.download_to_drive(temp_path)

        news_risk_text = build_news_risk_text(symbol)
        prompt = build_chart_prompt(caption, current_memory, news_risk_text)
        image_data_url = image_to_data_url(temp_path)
        reply = generate_vision_reply(prompt, image_data_url)

        await update.message.reply_text(reply)

        try:
            os.remove(temp_path)
        except Exception:
            pass

    except Exception as e:
        print("Vision Error:", e)
        error_text = str(e).lower()

        if "429" in error_text or "quota" in error_text or "rate" in error_text:
            await update.message.reply_text("目前图表分析人数较多，稍后再试一下。")
        elif "model" in error_text or "not found" in error_text:
            await update.message.reply_text("当前图表模型暂时不可用，稍后再试一下。")
        else:
            await update.message.reply_text("这张图我暂时没分析成功。你可以试试发更清晰的截图。")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()
    user_id = str(update.message.from_user.id)
    chat_id = update.message.chat_id

    current_memory = get_user_memory(user_id)
    intent = detect_user_intent(user_message)

    symbol = detect_symbol(user_message, current_memory)
    interval = detect_interval(user_message, current_memory)

    if user_message in ["我的设置", "设置"]:
        await show_settings(update, context)
        return

    if intent == "market_overview":
        try:
            reply = generate_market_overview_reply(user_message, current_memory)
        except Exception as e:
            print("Market Overview Error:", e)
            reply = "现在市场总览暂时读取不完整。简单说，这种时候先别急着重仓，等关键位和消息面确认会更稳。以上仅供行情参考，不构成投资建议。"

        await update.message.reply_text(reply)
        return

    lower = user_message.lower()

    if "今天" in user_message and ("数据" in user_message or "新闻" in user_message or "事件" in user_message):
        report = build_macro_report(days="today", symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "明天" in user_message and ("数据" in user_message or "新闻" in user_message or "事件" in user_message or "非农" in user_message):
        report = build_macro_report(days="tomorrow", symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "非农" in user_message or "nfp" in lower:
        report = build_macro_report(kind="nfp", days="week", symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "初请" in user_message or "失业金" in user_message or "jobless" in lower:
        report = build_macro_report(kind="jobless", days="week", symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "cpi" in lower or "通胀" in user_message:
        report = build_macro_report(kind="cpi", days="week", symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "fomc" in lower or "美联储" in user_message or "利率决议" in user_message:
        report = build_macro_report(kind="fomc", days="week", symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "订阅" in user_message and "提醒" in user_message:
        ok = subscribe_alert(user_id, chat_id, symbol)

        if ok:
            await update.message.reply_text(f"已开启 {symbol} 提醒，包含行情提醒和重要数据提醒。")
        else:
            await update.message.reply_text(f"你已经订阅过 {symbol} 提醒了。")

        return

    try:
        user_memory = update_user_memory(user_id, symbol, interval, user_message)
        news_risk_text = build_news_risk_text(symbol)

        if is_multi_timeframe_question(user_message):
            multi_tf_data = analyze_multi_timeframe(symbol)
            multi_tf_data = ensure_multi_tf_data(symbol, interval, multi_tf_data)
            summary = build_multi_timeframe_summary(multi_tf_data)

            if is_trading_plan_question(user_message):
                reply = generate_trade_plan_reply(
                    user_message,
                    symbol,
                    multi_tf_data,
                    summary,
                    user_memory,
                    news_risk_text
                )
            elif intent in ["why_move", "teaching", "risk_check", "macro_news", "general_market"]:
                reply = generate_flexible_market_reply(
                    user_message,
                    symbol,
                    multi_tf_data,
                    summary,
                    user_memory,
                    news_risk_text
                )
            else:
                reply = generate_ai_reply(
                    user_message,
                    symbol,
                    multi_tf_data,
                    summary,
                    user_memory,
                    news_risk_text
                )
        else:
            single_data = analyze_market(symbol, interval)

            summary = {
                "overall": single_data["trend"],
                "avg_long": single_data["long_probability"],
                "avg_short": single_data["short_probability"],
                "trend_text": f"{interval}：{single_data['trend']}"
            }

            multi_tf_data = {
                "15m": single_data,
                "1h": single_data,
                "4h": single_data
            }

            if is_trading_plan_question(user_message):
                reply = generate_trade_plan_reply(
                    user_message,
                    symbol,
                    multi_tf_data,
                    summary,
                    user_memory,
                    news_risk_text
                )
            elif intent in ["why_move", "teaching", "risk_check", "macro_news", "general_market"]:
                reply = generate_flexible_market_reply(
                    user_message,
                    symbol,
                    multi_tf_data,
                    summary,
                    user_memory,
                    news_risk_text
                )
            else:
                reply = generate_ai_reply(
                    user_message,
                    symbol,
                    multi_tf_data,
                    summary,
                    user_memory,
                    news_risk_text
                )

    except Exception as e:
        print("Error:", e)

        reply = """
目前行情系统暂时无法读取数据。

这位置先别急着追，等行情确认会更稳一些。

以上仅供行情参考，不构成投资建议。
"""

    await update.message.reply_text(reply)


def main():
    validate_env()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("tomorrow", tomorrow_command))
    app.add_handler(CommandHandler("nfp", nfp_command))
    app.add_handler(CommandHandler("cpi", cpi_command))
    app.add_handler(CommandHandler("jobless", jobless_command))
    app.add_handler(CommandHandler("fomc", fomc_command))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.job_queue.run_repeating(check_alerts, interval=300, first=30)
    app.job_queue.run_repeating(check_breaking_news, interval=180, first=45)

    print("V14 Flexible AI Trader 已启动...")
    print("API：OpenRouter")
    print("文字模型：", TEXT_MODEL_NAME)
    print("图片模型：", VISION_MODEL_NAME)
    print("宏观引擎：ForexFactory + 中文快讯")
    print("已开启：自然语言意图识别 + 市场总览 + 为什么涨跌解释 + 实时突发新闻推送 + AI交易计划A/B + Full Macro Engine + 多周期分析 + 图表识别")

    app.run_polling()


if __name__ == "__main__":
    main()
