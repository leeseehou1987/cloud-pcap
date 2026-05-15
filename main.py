import os
import json
import time
import base64
import tempfile
import re
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
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
GOLDAPI_KEY = os.getenv("GOLDAPI_KEY")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
TRADE_JOURNAL_FILE = "trade_journal.json"
MACRO_LIVE_STATE_FILE = "macro_live_state.json"
USER_IDEA_FILE = "user_ideas.json"
LEARNING_LOG_FILE = "learning_log.json"
MARKET_THOUGHT_FILE = "market_thoughts.json"

MULTI_TIMEFRAMES = ["15m", "1h", "4h"]

# =========================
# V31 Volatility Alert Engine
# =========================
VOL_ALERT_COOLDOWN = 300
VOL_ALERT_CACHE_FILE = "vol_alert_cache.json"


# =========================
# V33 Multi-Agent Trading Committee
# Macro + Technical + Liquidity + Psychology + Risk Agents
# =========================

def v33_vote_label(score):
    if score > 0:
        return "偏多"
    if score < 0:
        return "偏空"
    return "中性"


def v33_agent_result(name, score, confidence, reason, warning=""):
    return {
        "agent": name,
        "score": int(score),
        "vote": v33_vote_label(score),
        "confidence": int(clamp_value(confidence, 10, 95)) if "clamp_value" in globals() else int(max(10, min(95, confidence))),
        "reason": reason,
        "warning": warning
    }


def macro_agent(symbol, data, summary, news_risk_text=""):
    text = str(news_risk_text).lower()
    score = 0
    confidence = 50
    reasons = []

    if any(k in text for k in ["美元走弱", "美元下跌", "dxy down", "美元指数近几日变化：-"]):
        score += 2
        reasons.append("美元走弱对黄金/BTC偏支撑")
    if any(k in text for k in ["美元走强", "美元上涨", "dxy up"]):
        score -= 2
        reasons.append("美元走强通常压制黄金/BTC")

    if any(k in text for k in ["美债", "收益率"]):
        confidence += 8
        reasons.append("美债收益率变化会影响黄金和风险资产")

    if any(k in text for k in ["cpi", "非农", "fomc", "美联储", "利率", "pce", "ppi"]):
        confidence += 10
        reasons.append("存在重要宏观事件，宏观权重提高")

    if any(k in text for k in ["待公布", "pending", "等待公布", "数据源暂未更新"]):
        confidence -= 8
        reasons.append("部分数据仍待公布，不能过度确认方向")

    if is_gold_symbol(symbol):
        reason = "；".join(reasons) if reasons else "黄金主要受美元、美债、通胀预期和避险情绪影响"
    elif is_crypto_symbol(symbol):
        reason = "；".join(reasons) if reasons else "加密资产主要受美元、美债、风险情绪和监管消息影响"
    else:
        reason = "；".join(reasons) if reasons else "外部宏观信号暂时不够强，先以技术结构为主"

    return v33_agent_result("Macro Agent 宏观脑", score, confidence, reason)


def technical_agent(symbol, data, summary, news_risk_text=""):
    trend = data.get("trend", "震荡")
    rsi = safe_float(data.get("rsi", 50))
    structure = data.get("structure_event", "")
    avg_long = int(summary.get("avg_long", data.get("long_probability", 50)))
    avg_short = int(summary.get("avg_short", data.get("short_probability", 50)))

    score = 0
    confidence = 55
    reasons = []

    if trend == "偏多":
        score += 2
        reasons.append("短线趋势偏多")
    elif trend == "偏空":
        score -= 2
        reasons.append("短线趋势偏空")
    else:
        reasons.append("短线结构偏震荡")

    if avg_long - avg_short >= 15:
        score += 1
        confidence += 8
        reasons.append("多周期做多概率占优")
    elif avg_short - avg_long >= 15:
        score -= 1
        confidence += 8
        reasons.append("多周期做空概率占优")

    if "BOS 向上" in structure:
        score += 2
        confidence += 8
        reasons.append("出现向上BOS")
    elif "BOS 向下" in structure:
        score -= 2
        confidence += 8
        reasons.append("出现向下BOS")

    warning = ""
    if rsi >= 72:
        score -= 1
        warning = "RSI偏高，追多性价比下降"
    elif rsi <= 28:
        score += 1
        warning = "RSI偏低，追空性价比下降"

    return v33_agent_result("Technical Agent 技术脑", score, confidence, "；".join(reasons), warning)


def liquidity_agent(symbol, data, summary, news_risk_text=""):
    structure = data.get("structure_event", "")
    liquidity = data.get("liquidity", "")
    premium_discount = data.get("premium_discount", "")

    score = 0
    confidence = 55
    reasons = []

    if "扫高" in structure or "假突破" in structure:
        score -= 2
        confidence += 10
        reasons.append("上方扫高/假突破，容易诱多后回落")
    elif "扫低" in structure or "假跌破" in structure:
        score += 2
        confidence += 10
        reasons.append("下方扫低后收回，容易形成反弹")

    if "上方有等高流动性" in liquidity:
        reasons.append("上方存在止损/流动性池，可能先扫高")
    elif "下方有等低流动性" in liquidity:
        reasons.append("下方存在止损/流动性池，可能先扫低")

    if "区间偏高" in premium_discount:
        score -= 1
        reasons.append("价格处在区间偏高，追多风险提高")
    elif "区间偏低" in premium_discount:
        score += 1
        reasons.append("价格处在区间偏低，追空风险提高")

    if not reasons:
        reasons.append("暂时没有明显流动性扫盘信号")

    return v33_agent_result("Liquidity Agent 流动性脑", score, confidence, "；".join(reasons))


def psychology_agent(symbol, data, summary, news_risk_text=""):
    risk = data.get("risk", "")
    atr_pct = safe_float(data.get("atr_pct", 0))
    structure = data.get("structure_event", "")

    score = 0
    confidence = 50
    reasons = []

    if "追多风险" in risk or "超买" in risk:
        score -= 1
        confidence += 8
        reasons.append("市场可能有追多/FOMO情绪")
    elif "追空风险" in risk or "超卖" in risk:
        score += 1
        confidence += 8
        reasons.append("市场可能有恐慌追空情绪")

    if atr_pct >= 0.8:
        confidence += 10
        reasons.append("波动明显放大，情绪化交易概率上升")

    if "扫" in structure:
        confidence += 6
        reasons.append("扫流动性通常代表市场情绪被利用")

    if not reasons:
        reasons.append("情绪面暂时没有明显极端")

    return v33_agent_result("Psychology Agent 情绪脑", score, confidence, "；".join(reasons))


def risk_agent(symbol, data, summary, news_risk_text=""):
    text = str(news_risk_text).lower()
    atr_pct = safe_float(data.get("atr_pct", 0))
    risk = data.get("risk", "")

    score = 0
    confidence = 60
    reasons = []
    warning = ""

    if any(k in text for k in ["cpi", "非农", "fomc", "美联储", "利率", "pce", "ppi", "高影响"]):
        confidence += 12
        reasons.append("存在高影响宏观/新闻风险")
        warning = "不适合重仓，数据前后容易扫损"

    if any(k in text for k in ["待公布", "pending", "等待公布", "数据源暂未更新"]):
        confidence += 8
        reasons.append("关键数据仍有待公布/数据源未更新")
        warning = "方向判断需要打折"

    if atr_pct >= 0.8:
        confidence += 10
        reasons.append("ATR波动偏高")
        warning = "建议降低仓位，等待K线收稳"

    if "风险偏高" in risk:
        confidence += 6
        reasons.append(risk)

    # Risk Agent score is intentionally conservative.
    if reasons:
        score -= 1

    if not reasons:
        reasons.append("当前风控压力中等")

    return v33_agent_result("Risk Agent 风控脑", score, confidence, "；".join(reasons), warning)


def memory_agent(symbol, data, summary, news_risk_text=""):
    try:
        if "build_market_fingerprint" not in globals():
            return v33_agent_result("Memory Agent 记忆脑", 0, 40, "记忆模块暂未启用")

        fingerprint = build_market_fingerprint(symbol, data, summary, news_risk_text)
        similar = find_similar_brain_records("global", fingerprint, symbol=symbol, limit=5)
        wins = sum(1 for r in similar if r.get("outcome") == "win")
        losses = sum(1 for r in similar if r.get("outcome") == "loss")

        score = 0
        confidence = 45
        if wins + losses >= 2:
            win_rate = wins / max(wins + losses, 1)
            confidence += 15
            if win_rate >= 0.6:
                score += 1
            elif win_rate <= 0.4:
                score -= 1
            reason = f"找到 {len(similar)} 条类似样本，已复盘胜率约 {round(win_rate*100,1)}%"
        elif similar:
            reason = f"找到 {len(similar)} 条类似样本，但已复盘数量不足"
        else:
            reason = "暂无足够类似历史样本"
        return v33_agent_result("Memory Agent 记忆脑", score, confidence, reason)
    except Exception as e:
        return v33_agent_result("Memory Agent 记忆脑", 0, 35, f"记忆读取失败：{e}")


def run_v33_trading_committee(symbol, data, summary, news_risk_text=""):
    agents = [
        macro_agent(symbol, data, summary, news_risk_text),
        technical_agent(symbol, data, summary, news_risk_text),
        liquidity_agent(symbol, data, summary, news_risk_text),
        psychology_agent(symbol, data, summary, news_risk_text),
        risk_agent(symbol, data, summary, news_risk_text),
        memory_agent(symbol, data, summary, news_risk_text),
    ]

    bull_score = sum(max(a["score"], 0) * (a["confidence"] / 100) for a in agents)
    bear_score = sum(abs(min(a["score"], 0)) * (a["confidence"] / 100) for a in agents)

    raw = bull_score - bear_score
    risk_penalty = 0

    for a in agents:
        if "风控" in a["agent"] and a["score"] < 0:
            risk_penalty += 0.8
        if a.get("warning"):
            risk_penalty += 0.2

    if raw >= 1.5:
        final_bias = "谨慎偏多" if risk_penalty >= 1 else "偏多"
        direction = "long"
    elif raw <= -1.5:
        final_bias = "谨慎偏空" if risk_penalty >= 1 else "偏空"
        direction = "short"
    else:
        final_bias = "观望/震荡"
        direction = "neutral"

    confidence = int(max(35, min(88, 50 + abs(raw) * 12 - risk_penalty * 6)))

    warnings = [a["warning"] for a in agents if a.get("warning")]
    return {
        "agents": agents,
        "bull_score": round(bull_score, 2),
        "bear_score": round(bear_score, 2),
        "final_bias": final_bias,
        "direction": direction,
        "confidence": confidence,
        "warnings": warnings[:4],
    }


def build_v33_committee_context(symbol, data, summary, news_risk_text=""):
    result = run_v33_trading_committee(symbol, data, summary, news_risk_text)
    lines = ["【V33 AI交易委员会】"]

    for agent in result["agents"]:
        warning = f"｜提醒：{agent['warning']}" if agent.get("warning") else ""
        lines.append(
            f"{agent['agent']}：{agent['vote']}（信心{agent['confidence']}）｜{agent['reason']}{warning}"
        )

    lines.append("")
    lines.append(f"委员会投票：多头分 {result['bull_score']} / 空头分 {result['bear_score']}")
    lines.append(f"最终结论：{result['final_bias']}")
    lines.append(f"委员会信心：{result['confidence']} / 100")

    if result["warnings"]:
        lines.append("主要风险：" + " / ".join(result["warnings"]))

    lines.append("执行原则：如果委员会结论和风控脑冲突，以风控脑为优先。")
    return "\n".join(lines)


# =========================
# V32 AI Trading Brain
# =========================
TRADING_BRAIN_FILE = "trading_brain.json"
TRADING_BRAIN_MAX_RECORDS = 500
BRAIN_MIN_REVIEW_SECONDS = 1800

XAUUSD_1M_ALERT_USD = 12
XAUUSD_5M_ALERT_USD = 22

BTC_1M_ALERT_PCT = 1.8
BTC_5M_ALERT_PCT = 3.5

ATR_EXPLOSION_MULTIPLIER = 2.2

# V27 Macro State Engine
MACRO_RELEASE_LOOKBACK_MINUTES = 180
MACRO_PRE_RELEASE_WINDOW_MINUTES = 30
MACRO_POST_RELEASE_FORCE_REFRESH_MINUTES = 20

# =========================
# Timezone Config
# =========================
# Malaysia / Singapore / Hong Kong time
LOCAL_TIMEZONE = timezone(timedelta(hours=8))
LOCAL_TIMEZONE_NAME = "UTC+8"



def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_local_now():
    return datetime.now(LOCAL_TIMEZONE)


def format_local_time(dt=None):
    if dt is None:
        dt = get_local_now()

    return dt.strftime("%Y-%m-%d %H:%M UTC+8")


def get_time_context():
    return f"当前本地时间：{format_local_time()}。所有日期和时间请以 UTC+8 理解。"



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
    "黄金": "XAUUSD",
    "金价": "XAUUSD",
    "现货黄金": "XAUUSD",
    "伦敦金": "XAUUSD",
    "gold": "XAUUSD",
    "xau": "XAUUSD",
    "xauusd": "XAUUSD",

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


POSITION_SIZE_KEYWORDS = [
    "仓位", "算仓位", "仓位计算", "下多少", "开多少", "几手",
    "position size", "risk", "风险多少"
]

TRADE_JOURNAL_KEYWORDS = [
    "记录交易", "记一笔", "我做多", "我做空", "我进场",
    "我的交易日志", "交易日志", "复盘我的交易", "复盘"
]


SELF_LEARNING_KEYWORDS = [
    "记住我的想法", "记录我的想法", "我的想法是", "我的交易想法",
    "我的想法库", "删除想法", "清空想法",
    "学习今天行情", "总结今天行情", "今天复盘",
    "总结我的交易风格", "我的交易风格", "学习我的风格",
    "记录市场想法", "市场想法"
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
    "fomc": ["fomc", "federal reserve", "fed interest", "fed funds", "federal funds", "rate decision", "rate statement", "fomc statement", "fomc meeting minutes", "powell", "利率决议", "美联储", "鲍威尔"],
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

    "Empire State Manufacturing Index": "美国纽约联储制造业指数",
    "Capacity Utilization Rate": "美国产能利用率",
    "Industrial Production m/m": "美国工业生产月率",
    "Industrial Production y/y": "美国工业生产年率",
    "Manufacturing Production m/m": "美国制造业生产月率",
    "Business Inventories m/m": "美国商业库存月率",
    "NAHB Housing Market Index": "美国NAHB房产市场指数",
    "Building Permits": "美国营建许可",
    "Housing Starts": "美国新屋开工",
    "Existing Home Sales": "美国成屋销售",
    "New Home Sales": "美国新屋销售",
    "Pending Home Sales m/m": "美国成屋签约销售月率",
    "Philadelphia Fed Manufacturing Index": "美国费城联储制造业指数",
    "Richmond Manufacturing Index": "美国里士满联储制造业指数",
    "Chicago PMI": "美国芝加哥PMI",
    "CB Consumer Confidence": "美国谘商会消费者信心指数",
    "Prelim UoM Consumer Sentiment": "美国密歇根大学消费者信心指数初值",
    "Revised UoM Consumer Sentiment": "美国密歇根大学消费者信心指数终值",
    "UoM Consumer Sentiment": "美国密歇根大学消费者信心指数",
    "Prelim UoM Inflation Expectations": "美国密歇根大学通胀预期初值",
    "Revised UoM Inflation Expectations": "美国密歇根大学通胀预期终值",
    "JOLTS Job Openings": "美国JOLTS职位空缺",
    "ADP Non-Farm Employment Change": "美国ADP非农就业人数",
    "Challenger Job Cuts y/y": "美国挑战者企业裁员年率",
    "Average Hourly Earnings y/y": "美国平均时薪年率",
    "Labor Force Participation Rate": "美国劳动参与率",
    "ISM Manufacturing Prices": "美国ISM制造业物价指数",
    "ISM Manufacturing Employment": "美国ISM制造业就业指数",
    "ISM Services Prices": "美国ISM服务业物价指数",
    "ISM Services Employment": "美国ISM服务业就业指数",
    "S&P Global Manufacturing PMI": "美国标普全球制造业PMI",
    "S&P Global Services PMI": "美国标普全球服务业PMI",
    "S&P Global Composite PMI": "美国标普全球综合PMI",
    "Durable Goods Orders m/m": "美国耐用品订单月率",
    "Core Durable Goods Orders m/m": "美国核心耐用品订单月率",
    "Factory Orders m/m": "美国工厂订单月率",
    "Trade Balance": "美国贸易帐",
    "Goods Trade Balance": "美国商品贸易帐",
    "Import Prices m/m": "美国进口物价月率",
    "Export Prices m/m": "美国出口物价月率",
    "Wholesale Inventories m/m": "美国批发库存月率",
    "Crude Oil Inventories": "美国EIA原油库存",
    "Natural Gas Storage": "美国天然气库存",
    "Fed Chair Powell Testifies": "美联储主席鲍威尔作证",
    "FOMC Meeting Minutes": "FOMC会议纪要",
    "FOMC Economic Projections": "FOMC经济预测",
    "Federal Budget Balance": "美国联邦预算余额",
    "Treasury Currency Report": "美国财政部汇率报告",
    "Beige Book": "美联储褐皮书",
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
    
    "XAUUSD": ["黄金", "金价", "美元", "美债", "通胀", "美联储", "cpi", "pce", "避险"],
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
    "XAUUSD", "GC=F", "SI=F",
    "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "CAD=X", "CHF=X", "NZDUSD=X",
    "BTCUSDT", "ETHUSDT", "SOLUSDT"
]


SYSTEM_PROMPT = """
你不是分析报告机器人。
你是一个天天盯盘、说话像真人的交易员型 AI 助手。
你要像交易大师一样思考：先判断市场状态，再判断概率，再判断风险，最后给出清楚执行结论。必要时采用AI交易委员会思维：宏观、技术、流动性、情绪、风控、记忆分别判断，再综合。

最重要原则：
1. 必须先直接回答用户真正问的问题。
2. 不要一开口就讲指标。
3. 不要用户没问计划，就硬给计划A/计划B。
4. 用户问“能不能进/能不能追”，第一句必须直接说：我不太建议现在追 / 可以轻仓试但要止损 / 我会先等。
5. 用户问“为什么涨/跌”，第一句必须解释主因。
6. 用户问“怎么做/给计划”，才输出计划A/计划B。
7. 用户问教学，就用简单人话解释，不要硬扯入场点。

说话风格：
- 像真人交易员聊天
- 简短、有判断、有经验感
- 可以说“如果是我，我会先等”
- 可以说“这个位置我不太喜欢追”
- 可以说“这里容易两边扫”
- 不要像研究报告
- 不要机械列指标
- 当前价格优先使用实时价；如果价格源延迟，要提醒用户以交易平台实时报价为准

交易安全：
- 不保证涨跌
- 不说稳赚
- 不叫用户满仓、重仓、梭哈
- 只能做行情参考和风险提醒

如果有重要经济数据或突发事件：
- 必须先提醒消息面风险
- 数据前不建议重仓
- 建议等数据公布后 5~15 分钟，看方向稳定再判断

所有时间必须以 UTC+8 为准，不要自行猜测 UTC 日期。

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
    return symbol in ["XAUUSD", "GC=F"]


def is_silver_symbol(symbol):
    return symbol == "SI=F"


def is_forex_symbol(symbol):
    return symbol.endswith("=X")



def get_goldapi_spot_quote():
    """
    V26 Spot Gold:
    Fetch true spot gold XAU/USD from GoldAPI.
    Requires Railway/ENV variable:
    GOLDAPI_KEY=your_goldapi_key
    """
    if not GOLDAPI_KEY:
        return None

    try:
        url = "https://www.goldapi.io/api/XAU/USD"
        headers = {
            "x-access-token": GOLDAPI_KEY,
            "Content-Type": "application/json",
            "User-Agent": "AI-Trader-Bot/26.0"
        }

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code in [401, 403]:
            print("GoldAPI Auth Error: please check GOLDAPI_KEY")
            return None

        if response.status_code == 429:
            print("GoldAPI Rate Limit: using fallback price source")
            return None

        response.raise_for_status()
        data = response.json()

        price = data.get("price")
        bid = data.get("bid")
        ask = data.get("ask")

        if price is not None:
            return {
                "price": float(price),
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "source": data.get("symbol") or "GoldAPI XAU/USD",
                "timestamp": data.get("timestamp")
            }

        if bid is not None and ask is not None:
            return {
                "price": (float(bid) + float(ask)) / 2,
                "bid": float(bid),
                "ask": float(ask),
                "source": data.get("symbol") or "GoldAPI XAU/USD",
                "timestamp": data.get("timestamp")
            }

    except Exception as e:
        print("GoldAPI Error:", e)

    return None


def get_goldapi_spot_price():
    quote = get_goldapi_spot_quote()
    if quote and quote.get("price") is not None:
        return float(quote["price"])
    return None


def get_realtime_price(symbol):
    """
    V26 实时价格层：
    - 现货黄金 XAUUSD：GoldAPI spot XAU/USD
    - Crypto：Binance 实时 ticker
    - 黄金期货/外汇：Yahoo Finance 最近报价兜底
    注意：Yahoo 仍可能有轻微延迟；现货黄金优先使用 GoldAPI。
    """
    try:
        if symbol == "XAUUSD":
            spot_price = get_goldapi_spot_price()
            if spot_price:
                return float(spot_price)
            # Fallback to gold futures if GoldAPI is unavailable.
            symbol = "GC=F"

        if is_crypto_symbol(symbol):
            url = "https://data-api.binance.vision/api/v3/ticker/price"
            response = requests.get(url, params={"symbol": symbol}, timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data["price"])

        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)

        if fast_info:
            price = None

            try:
                price = fast_info.get("last_price")
            except Exception:
                try:
                    price = fast_info["last_price"]
                except Exception:
                    price = None

            if price:
                return float(price)

        hist = ticker.history(period="1d", interval="1m")

        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])

    except Exception as e:
        print("Realtime Price Error:", symbol, e)

    return None


def price_precision(symbol):
    if symbol.endswith("USDT"):
        return 2

    if symbol in ["XAUUSD", "GC=F", "SI=F"]:
        return 2

    if symbol.endswith("=X"):
        return 5

    return 4


def round_price(symbol, price):
    if price is None:
        return None

    return round(float(price), price_precision(symbol))


def get_asset_name(symbol):
    names = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "SOLUSDT": "SOL",
        "XAUUSD": "现货黄金",
        "GC=F": "黄金期货",
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



def detect_macro_time_range(user_message):
    text = normalize_text(user_message).lower()

    if any(k in text for k in ["前天", "2天前", "two days ago"]):
        return "2daysago"

    if any(k in text for k in ["昨天", "昨日", "yesterday"]):
        return "yesterday"

    if any(k in text for k in ["今天", "今日", "today"]):
        return "today"

    if any(k in text for k in ["明天", "tomorrow"]):
        return "tomorrow"

    if any(k in text for k in ["本周", "这周", "这一周", "week", "最近几天", "最近数据", "最近一次", "上一份", "上次"]):
        return "week"

    return "today_tomorrow"


def detect_macro_kind_from_text(user_message):
    text = normalize_text(user_message).lower()

    if "非农" in text or "nfp" in text or "payroll" in text:
        return "nfp"

    if "初请" in text or "失业金" in text or "jobless" in text:
        return "jobless"

    if "cpi" in text or "通胀" in text or "消费者物价" in text:
        return "cpi"

    if "fomc" in text or "美联储" in text or "利率决议" in text or "鲍威尔" in text:
        return "fomc"

    if "pce" in text:
        return "pce"

    if "ppi" in text or "生产者物价" in text:
        return "ppi"

    if "gdp" in text:
        return "gdp"

    if "零售" in text or "retail" in text:
        return "retail"

    if "pmi" in text or "ism" in text:
        return "pmi"

    return None


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
            "last_question": "",
            "user_id": user_id
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
            "last_question": "",
            "user_id": user_id
        }

    memory[user_id]["favorite_symbol"] = symbol
    memory[user_id]["favorite_interval"] = interval
    memory[user_id]["message_count"] += 1
    memory[user_id]["last_question"] = user_message
    memory[user_id]["user_id"] = user_id

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



def is_empty_macro_value(value):
    text = normalize_text(value)
    return text == "" or text.lower() in ["n/a", "na", "-", "none", "null", "等待公布", "待公布", "未公布"]


def macro_event_status(event):
    actual = event.get("actual", "")
    return "released" if not is_empty_macro_value(actual) else "pending"


def enrich_macro_event_state(event):
    event["status"] = macro_event_status(event)
    event["released"] = event["status"] == "released"
    if not event.get("event_id"):
        event["event_id"] = f"{event.get('time','')}_{event.get('country','')}_{event.get('title','')}"
    if not event.get("updated_at"):
        event["updated_at"] = format_local_time()
    return event


def macro_status_label(event):
    return "已公布" if macro_event_status(event) == "released" else "待公布"


def is_recent_macro_event(event, lookback_minutes=MACRO_RELEASE_LOOKBACK_MINUTES):
    try:
        event_dt = parse_event_local_datetime(event)
        if not event_dt:
            return True
        now = get_local_now()
        return now - timedelta(minutes=lookback_minutes) <= event_dt <= now + timedelta(days=7)
    except Exception:
        return True


def should_force_refresh_for_macro_question(events):
    now = get_local_now()
    for event in events:
        try:
            if not is_macro_high_impact(event):
                continue
            event_dt = parse_event_local_datetime(event)
            if not event_dt:
                continue
            minutes_diff = (now - event_dt).total_seconds() / 60
            if -MACRO_PRE_RELEASE_WINDOW_MINUTES <= minutes_diff <= MACRO_POST_RELEASE_FORCE_REFRESH_MINUTES:
                if macro_event_status(event) != "released":
                    return True
        except Exception:
            continue
    return False


def build_macro_state_context(events):
    if not events:
        return "宏观事件状态：暂无相关事件。"

    lines = ["宏观事件状态（AI必须严格按照status判断，不可以自行猜测）："]
    for event in events[:8]:
        event = enrich_macro_event_state(dict(event))
        status = event.get("status", "pending")
        status_cn = "已公布" if status == "released" else "待公布"
        actual = event.get("actual") or "暂无"
        forecast = event.get("forecast") or "暂无"
        previous = event.get("previous") or "暂无"
        lines.append(
            f"- {translate_country(event.get('country', ''))}｜{translate_macro_title(event.get('title', ''))}｜"
            f"时间:{event.get('time', '')}｜status:{status}（{status_cn}）｜"
            f"actual:{actual}｜forecast:{forecast}｜previous:{previous}｜source:{event.get('source', '')}"
        )
    lines.append("规则：只有 status=released 且 actual 有值，才可以说数据已经公布；否则必须说数据源暂未更新或仍显示待公布。")
    return "\n".join(lines)


def normalize_macro_event(raw):
    title = normalize_text(raw.get("title") or raw.get("event") or raw.get("name") or raw.get("Event"))
    country = normalize_text(raw.get("country") or raw.get("Country"))
    date = normalize_text(raw.get("date") or raw.get("Date"))
    time_value = normalize_text(raw.get("time") or raw.get("Time"))
    impact = normalize_text(raw.get("impact") or raw.get("Impact") or raw.get("importance") or raw.get("Importance"))
    actual = normalize_text(raw.get("actual") or raw.get("Actual"))
    forecast = normalize_text(raw.get("forecast") or raw.get("Forecast") or raw.get("consensus") or raw.get("Consensus"))
    previous = normalize_text(raw.get("previous") or raw.get("Previous"))
    status = normalize_text(raw.get("status") or "")
    if not status:
        status = "released" if not is_empty_macro_value(actual) else "pending"
    event_id = normalize_text(raw.get("event_id") or f"{time_value}_{country}_{title}")

    return {
        "event_id": event_id,
        "title": title,
        "country": country,
        "date": date,
        "time": time_value,
        "impact": impact,
        "actual": actual,
        "forecast": forecast,
        "previous": previous,
        "status": status,
        "released": status == "released",
        "source": normalize_text(raw.get("source") or "Macro Calendar"),
        "updated_at": format_local_time()
    }


def fetch_forexfactory_calendar(days="today", force_refresh=False):
    """
    V25 Quiet Macro Calendar Engine

    Fixes:
    - Avoids hammering ForexFactory after 429 / DNS failure
    - Uses a circuit breaker cooldown
    - Uses cache first
    - Falls back silently when source is unavailable
    - Reduces Railway log spam
    """
    cache = load_json(MACRO_CACHE_FILE, {})
    cache_key = f"forexfactory_{days}_{get_local_now().date().isoformat()}"
    now = time.time()

    # Fresh cache: use it directly.
    if not force_refresh and cache.get("key") == cache_key and now - cache.get("created_at", 0) < 1800:
        return cache.get("events", [])

    # Circuit breaker: if ForexFactory recently failed, do not keep retrying.
    last_fail_at = cache.get("last_fail_at", 0)
    if not force_refresh and last_fail_at and now - last_fail_at < 3600:
        cached_events = cache.get("events", [])
        if cached_events:
            return cached_events
        return []

    urls = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-Trader-Bot/25.0)",
        "Accept": "application/json,text/plain,*/*",
        "Connection": "close",
    }

    last_error = None

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)

            # 429 = rate limited. Stop immediately and cool down.
            if response.status_code == 429:
                raise RuntimeError("ForexFactory rate limited: 429 Too Many Requests")

            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list):
                raise ValueError("ForexFactory response is not a list")

            events = []
            today = get_local_now().date()
            target_dates = {today}

            if days == "tomorrow":
                target_dates = {today + timedelta(days=1)}
            elif days == "yesterday":
                target_dates = {today - timedelta(days=1)}
            elif days == "2daysago":
                target_dates = {today - timedelta(days=2)}
            elif days == "week":
                target_dates = {today + timedelta(days=i) for i in range(-3, 7)}
            elif days == "today_tomorrow":
                target_dates = {today, today + timedelta(days=1)}

            for item in data:
                if not isinstance(item, dict):
                    continue

                date_text = item.get("date", "")

                try:
                    event_dt = datetime.fromisoformat(str(date_text).replace("Z", "+00:00"))
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=timezone.utc)
                    event_dt = event_dt.astimezone(LOCAL_TIMEZONE)
                    event_date = event_dt.date()
                    time_value = event_dt.strftime("%Y-%m-%d %H:%M UTC+8")
                except Exception:
                    event_date = today
                    time_value = str(date_text)

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
                    "source": "ForexFactory/FairEconomy"
                }

                events.append(normalize_macro_event(raw))

            save_json(MACRO_CACHE_FILE, {
                "key": cache_key,
                "created_at": now,
                "events": events,
                "source_url": url,
                "last_success": format_local_time(),
                "last_fail_at": 0,
                "last_error": ""
            })

            return events

        except Exception as e:
            last_error = str(e)
            # Do not print every retry. One concise line is enough.
            print(f"ForexFactory V25 unavailable, using cache if possible: {last_error}")
            break

    cached_events = cache.get("events", [])
    save_json(MACRO_CACHE_FILE, {
        **cache,
        "last_fail_at": now,
        "last_error": last_error,
        "last_fail_time": format_local_time()
    })

    if cached_events:
        return cached_events

    return []

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


def filter_macro_events(kind=None, days="today_tomorrow", symbol=None, force_refresh=False):
    events = fetch_forexfactory_calendar(days=days, force_refresh=force_refresh)
    events = [enrich_macro_event_state(dict(event)) for event in events]

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

    # V29 fix:
    # Do NOT apply 180-minute freshness filter to historical queries.
    if days in ["today", "today_tomorrow", "tomorrow"]:
        events = [event for event in events if is_recent_macro_event(event)]

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
        return "当前数据源暂未返回实际值，因此状态仍按待公布处理；如果外部网站已公布，建议使用 /refreshmacro 强制刷新。数据前后波动可能会放大。"

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

    if not title:
        return ""

    # 1. Exact dictionary match
    if title in MACRO_TRANSLATION:
        return MACRO_TRANSLATION[title]

    lower_title = title.lower()

    # 2. Fuzzy dictionary match
    for en_title, zh_title in MACRO_TRANSLATION.items():
        if en_title.lower() in lower_title or lower_title in en_title.lower():
            return zh_title

    # 3. Keyword fallback rules
    keyword_rules = [
        (["empire state"], "美国纽约联储制造业指数"),
        (["capacity utilization"], "美国产能利用率"),
        (["industrial production"], "美国工业生产"),
        (["manufacturing production"], "美国制造业生产"),
        (["philadelphia fed"], "美国费城联储制造业指数"),
        (["richmond manufacturing"], "美国里士满联储制造业指数"),
        (["chicago pmi"], "美国芝加哥PMI"),
        (["consumer confidence"], "美国消费者信心指数"),
        (["uom", "michigan"], "美国密歇根大学消费者信心指数"),
        (["inflation expectations"], "美国通胀预期"),
        (["jolts"], "美国JOLTS职位空缺"),
        (["adp"], "美国ADP非农就业人数"),
        (["non-farm", "nonfarm", "payroll"], "美国非农就业人数"),
        (["initial jobless", "jobless claims"], "美国初请失业金人数"),
        (["continuing jobless"], "美国续请失业金人数"),
        (["unemployment rate"], "美国失业率"),
        (["average hourly earnings"], "美国平均时薪"),
        (["core cpi"], "美国核心CPI通胀数据"),
        (["cpi", "consumer price"], "美国CPI通胀数据"),
        (["core pce"], "美国核心PCE物价指数"),
        (["pce"], "美国PCE物价指数"),
        (["ppi", "producer price"], "美国PPI生产者物价指数"),
        (["retail sales"], "美国零售销售"),
        (["durable goods"], "美国耐用品订单"),
        (["factory orders"], "美国工厂订单"),
        (["building permits"], "美国营建许可"),
        (["housing starts"], "美国新屋开工"),
        (["existing home sales"], "美国成屋销售"),
        (["new home sales"], "美国新屋销售"),
        (["pending home sales"], "美国成屋签约销售"),
        (["ism manufacturing"], "美国ISM制造业PMI"),
        (["ism services"], "美国ISM服务业PMI"),
        (["manufacturing pmi"], "制造业PMI"),
        (["services pmi"], "服务业PMI"),
        (["composite pmi"], "综合PMI"),
        (["federal funds", "rate decision"], "美联储利率决议"),
        (["fomc statement"], "FOMC政策声明"),
        (["fomc meeting minutes"], "FOMC会议纪要"),
        (["fomc"], "FOMC美联储会议"),
        (["powell"], "美联储主席鲍威尔讲话"),
        (["gdp"], "美国GDP数据"),
        (["trade balance"], "美国贸易帐"),
        (["crude oil inventories"], "美国EIA原油库存"),
        (["natural gas storage"], "美国天然气库存"),
    ]

    for keywords, zh_title in keyword_rules:
        if any(keyword in lower_title for keyword in keywords):
            return zh_title

    # 4. If not recognized, keep original English title.
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



# =========================
# V20 Macro Live Engine
# =========================

def is_empty_actual(value):
    text = normalize_text(value)
    return text == "" or text.lower() in ["n/a", "na", "-", "none", "null", "等待公布"]


def parse_event_local_datetime(event):
    time_text = normalize_text(event.get("time"))

    for fmt in ["%Y-%m-%d %H:%M UTC+8", "%Y-%m-%d %H:%M UTC"]:
        try:
            dt = datetime.strptime(time_text, fmt)
            return dt.replace(tzinfo=LOCAL_TIMEZONE)
        except Exception:
            pass

    date_text = normalize_text(event.get("date"))
    if date_text:
        try:
            dt = datetime.strptime(date_text, "%Y-%m-%d")
            return dt.replace(tzinfo=LOCAL_TIMEZONE)
        except Exception:
            pass

    return None


def is_event_due_for_refresh(event, grace_minutes=1):
    event_dt = parse_event_local_datetime(event)

    if not event_dt:
        return False

    return get_local_now() >= event_dt + timedelta(minutes=grace_minutes)


def should_force_macro_refresh(events):
    for event in events:
        if is_macro_high_impact(event) and is_empty_actual(event.get("actual")) and is_event_due_for_refresh(event):
            return True

    return False


def get_macro_events_live(kind=None, days="today_tomorrow", symbol=None):
    events = filter_macro_events(kind=kind, days=days, symbol=symbol, force_refresh=False)

    if should_force_macro_refresh(events) or should_force_refresh_for_macro_question(events):
        events = filter_macro_events(kind=kind, days=days, symbol=symbol, force_refresh=True)

    return [enrich_macro_event_state(dict(event)) for event in events]


def macro_surprise_text(event):
    title = event.get("title", "")
    actual = event.get("actual", "")
    forecast = event.get("forecast", "")

    actual_num = parse_macro_value(actual)
    forecast_num = parse_macro_value(forecast)

    if actual_num is None or forecast_num is None:
        return "实际值已公布，但暂时无法和预测做精确比较。"

    lower_title = title.lower()

    if actual_num > forecast_num:
        direction = "高于预期"
    elif actual_num < forecast_num:
        direction = "低于预期"
    else:
        direction = "符合预期"

    if any(word in lower_title for word in ["cpi", "pce", "ppi", "inflation", "price"]):
        if actual_num > forecast_num:
            impact = "通胀偏热，通常利多美元，黄金/BTC 可能承压。"
        elif actual_num < forecast_num:
            impact = "通胀降温，通常利空美元，黄金/BTC 可能获得支撑。"
        else:
            impact = "基本符合预期，市场可能转去看细项和美联储预期。"
        return f"实际值{direction}。{impact}"

    if any(word in lower_title for word in ["jobless", "unemployment", "失业"]):
        if actual_num > forecast_num:
            impact = "就业偏弱，通常利空美元，黄金/BTC 可能获得支撑。"
        elif actual_num < forecast_num:
            impact = "就业偏强，通常利多美元，黄金/BTC 可能承压。"
        else:
            impact = "基本符合预期，市场反应可能不会太单边。"
        return f"实际值{direction}。{impact}"

    if any(word in lower_title for word in ["payroll", "employment", "non-farm"]):
        if actual_num > forecast_num:
            impact = "就业强劲，通常利多美元，黄金/BTC 可能承压。"
        elif actual_num < forecast_num:
            impact = "就业走弱，通常利空美元，黄金/BTC 可能获得支撑。"
        else:
            impact = "基本符合预期，市场可能关注失业率和薪资细项。"
        return f"实际值{direction}。{impact}"

    return f"实际值{direction}，短线可能放大波动，需要看美元和美债反应。"


def build_macro_release_message(event):
    country = translate_country(event.get("country", ""))
    title = translate_macro_title(event.get("title", ""))

    return f"""
【重要数据公布】

{country}｜{title}

时间：{event.get('time', '')}
前值：{event.get('previous', '') or '暂无'}
市场预测：{event.get('forecast', '') or '暂无'}
实际值：{event.get('actual', '') or '暂无'}

解读：
{macro_surprise_text(event)}

提醒：
数据公布后 5~15 分钟容易来回扫，先看价格是否真正站稳关键位。

以上仅供行情参考，不构成投资建议。
""".strip()


def macro_release_key(event):
    return f"{event.get('time', '')}_{event.get('country', '')}_{event.get('title', '')}_{event.get('actual', '')}"


async def check_macro_live_releases(context: ContextTypes.DEFAULT_TYPE):
    alerts = load_json(ALERT_FILE, [])

    if not alerts:
        return

    state = load_json(MACRO_LIVE_STATE_FILE, {"sent_keys": []})
    sent_keys = set(state.get("sent_keys", []))

    try:
        # V25: do not force refresh every 60 seconds.
        # ForexFactory rate-limits easily; normal fetch uses cache/circuit breaker.
        events = fetch_forexfactory_calendar(days="today", force_refresh=False)
    except Exception as e:
        print("Macro Live Fetch Error:", e)
        return

    candidate_events = []

    for event in events:
        if not is_macro_high_impact(event):
            continue

        if is_empty_actual(event.get("actual")):
            continue

        event_dt = parse_event_local_datetime(event)
        if event_dt and get_local_now() < event_dt - timedelta(minutes=5):
            continue

        key = macro_release_key(event)

        if key in sent_keys:
            continue

        candidate_events.append((key, event))

    if not candidate_events:
        return

    unique_chat_ids = set()

    for item in alerts:
        chat_id = item.get("chat_id")
        if chat_id:
            unique_chat_ids.add(chat_id)

    new_keys = []

    for key, event in candidate_events[:5]:
        message = build_macro_release_message(event)

        for chat_id in unique_chat_ids:
            try:
                await context.bot.send_message(chat_id=chat_id, text=message)
            except Exception as e:
                print("Macro Live Send Error:", e)

        new_keys.append(key)

    all_keys = list(sent_keys) + new_keys
    all_keys = all_keys[-300:]

    save_json(MACRO_LIVE_STATE_FILE, {"sent_keys": all_keys, "last_check": int(time.time())})



def format_macro_event(event):
    event = enrich_macro_event_state(dict(event))
    country = translate_country(event.get("country", ""))
    title = translate_macro_title(event.get("title", ""))
    impact = translate_impact(event.get("impact", ""))
    status_cn = macro_status_label(event)
    actual_text = event.get("actual", "")
    if is_empty_macro_value(actual_text):
        actual_text = "等待公布 / 数据源暂未更新"

    return f"""
{country}｜{title}

时间：{event.get('time', '')}
状态：{status_cn}
影响级别：{impact}
前值：{event.get('previous', '') or '暂无'}
市场预测：{event.get('forecast', '') or '暂无'}
实际值：{actual_text}
市场解读：{explain_macro_event(event)}
""".strip()


def build_macro_report(kind=None, days="today_tomorrow", symbol=None):
    events = get_macro_events_live(kind=kind, days=days, symbol=symbol)

    if should_force_refresh_for_macro_question(events):
        events = filter_macro_events(kind=kind, days=days, symbol=symbol, force_refresh=True)

    if not events:
        return "暂时没有找到相关经济数据。"

    important = [event for event in events if is_macro_high_impact(event)]
    selected = important if important else events

    blocks = [format_macro_event(event) for event in selected[:8]]

    return "\n\n".join(blocks)



def build_macro_report_from_message(user_message, symbol=None):
    days = detect_macro_time_range(user_message)
    kind = detect_macro_kind_from_text(user_message)

    report = build_macro_report(kind=kind, days=days, symbol=symbol)

    # If asking "最近一次/上次" and exact day result is empty, widen to week.
    if "暂时没有找到相关经济数据" in report and any(k in user_message for k in ["最近一次", "上一份", "上次"]):
        report = build_macro_report(kind=kind, days="week", symbol=symbol)

    # V29: If user asks a specific historical event but none found,
    # show all macro events for that date instead of a dead answer.
    if "暂时没有找到相关经济数据" in report and days in ["yesterday", "2daysago"] and kind:
        all_report = build_macro_report(kind=None, days=days, symbol=symbol)
        if "暂时没有找到相关经济数据" not in all_report:
            event_name = {
                "fomc": "FOMC/美联储",
                "cpi": "CPI",
                "nfp": "非农",
                "jobless": "初请失业金",
                "pce": "PCE",
                "ppi": "PPI",
                "gdp": "GDP",
                "retail": "零售销售",
                "pmi": "PMI",
            }.get(kind, kind)
            report = f"昨天没有找到 {event_name} 的相关记录。下面是该日期可读取到的其他经济数据:\n\n{all_report}"

    return report


def get_macro_risk(symbol):
    events = get_macro_events_live(days="today_tomorrow", symbol=symbol)
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
            f"状态:{macro_status_label(event)}｜前值:{event.get('previous', '') or '暂无'}｜市场预测:{event.get('forecast', '') or '暂无'}｜实际值:{event.get('actual', '') or '待公布/数据源暂未更新'}"
        )

    return {
        "has_risk": True,
        "summary": "\n".join(lines),
        "events": selected
    }


def build_news_risk_text(symbol):
    chinese_news_text = build_chinese_news_text(symbol)
    macro_risk = get_macro_risk(symbol)
    macro_state_context = build_macro_state_context(macro_risk.get("events", []))

    if not macro_risk["has_risk"]:
        macro_text = macro_risk["summary"]
    else:
        macro_text = f"""
未来 24~48 小时检测到可能影响行情的重要经济数据：

{macro_risk['summary']}\n\n{macro_state_context}\n\n宏观风控：
数据公布前后 5~15 分钟波动可能放大，不建议提前重仓进场。
如果 actual 已公布且和预测差距较大，黄金、美元、BTC、外汇都可能快速波动。\n如果 status 仍是 pending，不要说数据已经公布，只能说数据源暂未更新。
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
    # V26: XAUUSD spot real-time price uses GoldAPI.
    # For technical candles, fallback to GC=F until a dedicated spot-gold OHLC API is added.
    if symbol == "XAUUSD":
        symbol = "GC=F"

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

    kline_close_price = float(close.iloc[-1])
    realtime_price = get_realtime_price(symbol)
    current_price = float(realtime_price) if realtime_price else kline_close_price

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

    # V22 Intelligence Metrics
    atr_pct = (atr / max(current_price, 0.0001)) * 100
    range_pct = ((resistance - support) / max(current_price, 0.0001)) * 100

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
        "price": round_price(symbol, current_price),
        "kline_close": round_price(symbol, kline_close_price),
        "realtime_price": round_price(symbol, realtime_price) if realtime_price else None,
        "price_source": "实时价" if realtime_price else "K线收盘价",
        "ma20": round(float(ma20), 4),
        "ma50": round(float(ma50), 4),
        "ema20": round(float(ema20), 4),
        "ema50": round(float(ema50), 4),
        "rsi": round(float(rsi), 2),
        "atr": round(float(atr), 4),
        "atr_pct": round(float(atr_pct), 3),
        "range_pct": round(float(range_pct), 3),
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




# =========================
# V22 Intelligence Layer
# Market Regime + Scenario Engine + Confidence + Macro Linkage
# =========================

def detect_market_regime(symbol, data, summary=None, news_risk_text=""):
    """
    先判断市场环境，再决定要不要追、等回踩、观望或降仓。
    这比单纯 RSI/MACD 更接近真人交易员的判断顺序。
    """
    trend = data.get("trend", "震荡")
    atr_pct = float(data.get("atr_pct", 0) or 0)
    range_pct = float(data.get("range_pct", 0) or 0)
    structure = data.get("structure_event", "")
    risk = data.get("risk", "")
    news_text = str(news_risk_text).lower()

    has_macro_risk = any(k in news_text for k in [
        "cpi", "pce", "非农", "fomc", "美联储", "利率", "初请", "powell", "鲍威尔", "高影响", "待公布"
    ])

    if has_macro_risk and atr_pct >= 0.25:
        return {
            "regime": "news_volatility",
            "label": "消息波动市",
            "action": "数据/消息风险偏高，优先降仓或观望，等 5~15 分钟确认方向。",
            "avoid": "避免第一根大阳/大阴追单。"
        }

    if "扫" in structure or "假" in structure:
        return {
            "regime": "liquidity_sweep",
            "label": "扫流动性行情",
            "action": "优先等扫高/扫低后的收回确认，不要追突破第一下。",
            "avoid": "避免在流动性刚被扫完时追单。"
        }

    if trend in ["偏多", "偏空"] and summary:
        avg_long = int(summary.get("avg_long", 50))
        avg_short = int(summary.get("avg_short", 50))
        if max(avg_long, avg_short) >= 68:
            return {
                "regime": "trend_continuation",
                "label": "趋势延续市",
                "action": "顺势思路优先，但更适合等回踩/反弹确认，不适合中间价硬追。",
                "avoid": "避免追在压力/支撑附近。"
            }

    if range_pct <= 1.2 or trend == "震荡":
        return {
            "regime": "range_market",
            "label": "震荡区间市",
            "action": "低吸高抛思路优先，区间中间位置少做。",
            "avoid": "避免把震荡误判成单边趋势。"
        }

    if atr_pct >= 0.8:
        return {
            "regime": "high_volatility",
            "label": "高波动市",
            "action": "止损要放宽、仓位要降低，等K线收稳再判断。",
            "avoid": "避免用平时仓位交易。"
        }

    return {
        "regime": "normal_market",
        "label": "普通行情",
        "action": "按关键位和多周期方向判断，等待确认比预测更重要。",
        "avoid": "避免没有触发条件就提前进场。"
    }


def v22_recent_change(symbol, period="5d", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 2:
            return None
        return round(((float(close.iloc[-1]) - float(close.iloc[0])) / max(float(close.iloc[0]), 0.0001)) * 100, 2)
    except Exception as e:
        print("V22 Recent Change Error:", symbol, e)
        return None


def build_macro_linkage_context(symbol):
    """
    简单宏观联动层：不是预测，只是提醒当前品种最该盯哪些外部变量。
    """
    dxy = v22_recent_change("DX-Y.NYB")
    us10y = v22_recent_change("^TNX")
    nq = v22_recent_change("NQ=F")

    parts = []
    if dxy is not None:
        parts.append(f"美元指数近几日变化：{dxy}%")
    if us10y is not None:
        parts.append(f"美债10年收益率近几日变化：{us10y}%")
    if nq is not None:
        parts.append(f"纳指期货近几日变化：{nq}%")

    if is_gold_symbol(symbol):
        focus = "黄金重点看美元和美债：美元/美债走强通常压制黄金，走弱通常支撑黄金。"
    elif is_crypto_symbol(symbol):
        focus = "BTC/ETH 重点看风险资产情绪、美元和美债：纳指强且美元弱时更容易支撑加密资产。"
    elif is_forex_symbol(symbol):
        focus = "外汇重点看美元强弱、对应央行预期和美债收益率变化。"
    else:
        focus = "该品种需要结合美元、美债和风险情绪一起看。"

    if not parts:
        return focus + " 当前外部联动数据暂时读取不完整。"

    return focus + "\n" + "\n".join(parts)


def calculate_v22_confidence(data, summary, regime_info, news_risk_text=""):
    score = 50
    avg_long = int(summary.get("avg_long", 50))
    avg_short = int(summary.get("avg_short", 50))
    direction_strength = abs(avg_long - avg_short)

    score += min(direction_strength // 2, 20)

    if data.get("trend") != "震荡":
        score += 8

    if "BOS" in data.get("structure_event", ""):
        score += 8

    if regime_info.get("regime") == "trend_continuation":
        score += 8
    elif regime_info.get("regime") in ["news_volatility", "liquidity_sweep", "high_volatility"]:
        score -= 15
    elif regime_info.get("regime") == "range_market":
        score -= 5

    if any(k in str(news_risk_text).lower() for k in ["cpi", "非农", "fomc", "美联储", "待公布", "高影响"]):
        score -= 10

    score = max(10, min(90, int(score)))

    if score >= 70:
        label = "较高"
    elif score >= 55:
        label = "中等"
    else:
        label = "偏低"

    return {"score": score, "label": label}


def build_scenario_engine(symbol, data, summary, regime_info, news_risk_text=""):
    asset = get_asset_name(symbol)
    support = data.get("support")
    resistance = data.get("resistance")
    trend = data.get("trend")

    if regime_info.get("regime") == "news_volatility":
        return f"""
情景推演：
A. 如果数据/消息利多并站稳 {resistance} 上方，{asset} 才更像继续走强。
B. 如果消息后冲高回落并跌回 {resistance} 下方，要小心假突破。
C. 如果跌破 {support} 后不能快速收回，短线会转弱。""".strip()

    if trend == "偏多":
        return f"""
情景推演：
A. 站稳 {resistance} 上方，多头延续概率提高。
B. 回踩不破 {support}，更像健康回调。
C. 跌破 {support} 且反抽失败，多头思路先失效。""".strip()

    if trend == "偏空":
        return f"""
情景推演：
A. 跌破 {support} 下方，空头延续概率提高。
B. 反弹不过 {resistance}，更像弱反抽。
C. 突破 {resistance} 并站稳，空头思路先失效。""".strip()

    return f"""
情景推演：
A. 区间内靠近 {support} 看止跌确认，不追空。
B. 靠近 {resistance} 看受压确认，不追多。
C. 真正站稳区间外，再考虑顺势跟随。""".strip()


def build_v22_intelligence_context(symbol, data, summary, news_risk_text):
    regime_info = detect_market_regime(symbol, data, summary, news_risk_text)
    confidence = calculate_v22_confidence(data, summary, regime_info, news_risk_text)
    scenarios = build_scenario_engine(symbol, data, summary, regime_info, news_risk_text)
    linkage = build_macro_linkage_context(symbol)

    return f"""
【V22智能判断层】
市场状态：{regime_info['label']}
当前处理方式：{regime_info['action']}
需要避免：{regime_info['avoid']}
AI判断置信度：{confidence['score']} / 100（{confidence['label']}）

宏观/关联市场：
{linkage}

{scenarios}
""".strip()





# =========================
# V33 Multi-Agent Trading Committee
# Macro + Technical + Liquidity + Psychology + Risk Agents
# =========================

def v33_vote_label(score):
    if score > 0:
        return "偏多"
    if score < 0:
        return "偏空"
    return "中性"


def v33_agent_result(name, score, confidence, reason, warning=""):
    return {
        "agent": name,
        "score": int(score),
        "vote": v33_vote_label(score),
        "confidence": int(clamp_value(confidence, 10, 95)) if "clamp_value" in globals() else int(max(10, min(95, confidence))),
        "reason": reason,
        "warning": warning
    }


def macro_agent(symbol, data, summary, news_risk_text=""):
    text = str(news_risk_text).lower()
    score = 0
    confidence = 50
    reasons = []

    if any(k in text for k in ["美元走弱", "美元下跌", "dxy down", "美元指数近几日变化：-"]):
        score += 2
        reasons.append("美元走弱对黄金/BTC偏支撑")
    if any(k in text for k in ["美元走强", "美元上涨", "dxy up"]):
        score -= 2
        reasons.append("美元走强通常压制黄金/BTC")

    if any(k in text for k in ["美债", "收益率"]):
        confidence += 8
        reasons.append("美债收益率变化会影响黄金和风险资产")

    if any(k in text for k in ["cpi", "非农", "fomc", "美联储", "利率", "pce", "ppi"]):
        confidence += 10
        reasons.append("存在重要宏观事件，宏观权重提高")

    if any(k in text for k in ["待公布", "pending", "等待公布", "数据源暂未更新"]):
        confidence -= 8
        reasons.append("部分数据仍待公布，不能过度确认方向")

    if is_gold_symbol(symbol):
        reason = "；".join(reasons) if reasons else "黄金主要受美元、美债、通胀预期和避险情绪影响"
    elif is_crypto_symbol(symbol):
        reason = "；".join(reasons) if reasons else "加密资产主要受美元、美债、风险情绪和监管消息影响"
    else:
        reason = "；".join(reasons) if reasons else "外部宏观信号暂时不够强，先以技术结构为主"

    return v33_agent_result("Macro Agent 宏观脑", score, confidence, reason)


def technical_agent(symbol, data, summary, news_risk_text=""):
    trend = data.get("trend", "震荡")
    rsi = safe_float(data.get("rsi", 50))
    structure = data.get("structure_event", "")
    avg_long = int(summary.get("avg_long", data.get("long_probability", 50)))
    avg_short = int(summary.get("avg_short", data.get("short_probability", 50)))

    score = 0
    confidence = 55
    reasons = []

    if trend == "偏多":
        score += 2
        reasons.append("短线趋势偏多")
    elif trend == "偏空":
        score -= 2
        reasons.append("短线趋势偏空")
    else:
        reasons.append("短线结构偏震荡")

    if avg_long - avg_short >= 15:
        score += 1
        confidence += 8
        reasons.append("多周期做多概率占优")
    elif avg_short - avg_long >= 15:
        score -= 1
        confidence += 8
        reasons.append("多周期做空概率占优")

    if "BOS 向上" in structure:
        score += 2
        confidence += 8
        reasons.append("出现向上BOS")
    elif "BOS 向下" in structure:
        score -= 2
        confidence += 8
        reasons.append("出现向下BOS")

    warning = ""
    if rsi >= 72:
        score -= 1
        warning = "RSI偏高，追多性价比下降"
    elif rsi <= 28:
        score += 1
        warning = "RSI偏低，追空性价比下降"

    return v33_agent_result("Technical Agent 技术脑", score, confidence, "；".join(reasons), warning)


def liquidity_agent(symbol, data, summary, news_risk_text=""):
    structure = data.get("structure_event", "")
    liquidity = data.get("liquidity", "")
    premium_discount = data.get("premium_discount", "")

    score = 0
    confidence = 55
    reasons = []

    if "扫高" in structure or "假突破" in structure:
        score -= 2
        confidence += 10
        reasons.append("上方扫高/假突破，容易诱多后回落")
    elif "扫低" in structure or "假跌破" in structure:
        score += 2
        confidence += 10
        reasons.append("下方扫低后收回，容易形成反弹")

    if "上方有等高流动性" in liquidity:
        reasons.append("上方存在止损/流动性池，可能先扫高")
    elif "下方有等低流动性" in liquidity:
        reasons.append("下方存在止损/流动性池，可能先扫低")

    if "区间偏高" in premium_discount:
        score -= 1
        reasons.append("价格处在区间偏高，追多风险提高")
    elif "区间偏低" in premium_discount:
        score += 1
        reasons.append("价格处在区间偏低，追空风险提高")

    if not reasons:
        reasons.append("暂时没有明显流动性扫盘信号")

    return v33_agent_result("Liquidity Agent 流动性脑", score, confidence, "；".join(reasons))


def psychology_agent(symbol, data, summary, news_risk_text=""):
    risk = data.get("risk", "")
    atr_pct = safe_float(data.get("atr_pct", 0))
    structure = data.get("structure_event", "")

    score = 0
    confidence = 50
    reasons = []

    if "追多风险" in risk or "超买" in risk:
        score -= 1
        confidence += 8
        reasons.append("市场可能有追多/FOMO情绪")
    elif "追空风险" in risk or "超卖" in risk:
        score += 1
        confidence += 8
        reasons.append("市场可能有恐慌追空情绪")

    if atr_pct >= 0.8:
        confidence += 10
        reasons.append("波动明显放大，情绪化交易概率上升")

    if "扫" in structure:
        confidence += 6
        reasons.append("扫流动性通常代表市场情绪被利用")

    if not reasons:
        reasons.append("情绪面暂时没有明显极端")

    return v33_agent_result("Psychology Agent 情绪脑", score, confidence, "；".join(reasons))


def risk_agent(symbol, data, summary, news_risk_text=""):
    text = str(news_risk_text).lower()
    atr_pct = safe_float(data.get("atr_pct", 0))
    risk = data.get("risk", "")

    score = 0
    confidence = 60
    reasons = []
    warning = ""

    if any(k in text for k in ["cpi", "非农", "fomc", "美联储", "利率", "pce", "ppi", "高影响"]):
        confidence += 12
        reasons.append("存在高影响宏观/新闻风险")
        warning = "不适合重仓，数据前后容易扫损"

    if any(k in text for k in ["待公布", "pending", "等待公布", "数据源暂未更新"]):
        confidence += 8
        reasons.append("关键数据仍有待公布/数据源未更新")
        warning = "方向判断需要打折"

    if atr_pct >= 0.8:
        confidence += 10
        reasons.append("ATR波动偏高")
        warning = "建议降低仓位，等待K线收稳"

    if "风险偏高" in risk:
        confidence += 6
        reasons.append(risk)

    # Risk Agent score is intentionally conservative.
    if reasons:
        score -= 1

    if not reasons:
        reasons.append("当前风控压力中等")

    return v33_agent_result("Risk Agent 风控脑", score, confidence, "；".join(reasons), warning)


def memory_agent(symbol, data, summary, news_risk_text=""):
    try:
        if "build_market_fingerprint" not in globals():
            return v33_agent_result("Memory Agent 记忆脑", 0, 40, "记忆模块暂未启用")

        fingerprint = build_market_fingerprint(symbol, data, summary, news_risk_text)
        similar = find_similar_brain_records("global", fingerprint, symbol=symbol, limit=5)
        wins = sum(1 for r in similar if r.get("outcome") == "win")
        losses = sum(1 for r in similar if r.get("outcome") == "loss")

        score = 0
        confidence = 45
        if wins + losses >= 2:
            win_rate = wins / max(wins + losses, 1)
            confidence += 15
            if win_rate >= 0.6:
                score += 1
            elif win_rate <= 0.4:
                score -= 1
            reason = f"找到 {len(similar)} 条类似样本，已复盘胜率约 {round(win_rate*100,1)}%"
        elif similar:
            reason = f"找到 {len(similar)} 条类似样本，但已复盘数量不足"
        else:
            reason = "暂无足够类似历史样本"
        return v33_agent_result("Memory Agent 记忆脑", score, confidence, reason)
    except Exception as e:
        return v33_agent_result("Memory Agent 记忆脑", 0, 35, f"记忆读取失败：{e}")


def run_v33_trading_committee(symbol, data, summary, news_risk_text=""):
    agents = [
        macro_agent(symbol, data, summary, news_risk_text),
        technical_agent(symbol, data, summary, news_risk_text),
        liquidity_agent(symbol, data, summary, news_risk_text),
        psychology_agent(symbol, data, summary, news_risk_text),
        risk_agent(symbol, data, summary, news_risk_text),
        memory_agent(symbol, data, summary, news_risk_text),
    ]

    bull_score = sum(max(a["score"], 0) * (a["confidence"] / 100) for a in agents)
    bear_score = sum(abs(min(a["score"], 0)) * (a["confidence"] / 100) for a in agents)

    raw = bull_score - bear_score
    risk_penalty = 0

    for a in agents:
        if "风控" in a["agent"] and a["score"] < 0:
            risk_penalty += 0.8
        if a.get("warning"):
            risk_penalty += 0.2

    if raw >= 1.5:
        final_bias = "谨慎偏多" if risk_penalty >= 1 else "偏多"
        direction = "long"
    elif raw <= -1.5:
        final_bias = "谨慎偏空" if risk_penalty >= 1 else "偏空"
        direction = "short"
    else:
        final_bias = "观望/震荡"
        direction = "neutral"

    confidence = int(max(35, min(88, 50 + abs(raw) * 12 - risk_penalty * 6)))

    warnings = [a["warning"] for a in agents if a.get("warning")]
    return {
        "agents": agents,
        "bull_score": round(bull_score, 2),
        "bear_score": round(bear_score, 2),
        "final_bias": final_bias,
        "direction": direction,
        "confidence": confidence,
        "warnings": warnings[:4],
    }


def build_v33_committee_context(symbol, data, summary, news_risk_text=""):
    result = run_v33_trading_committee(symbol, data, summary, news_risk_text)
    lines = ["【V33 AI交易委员会】"]

    for agent in result["agents"]:
        warning = f"｜提醒：{agent['warning']}" if agent.get("warning") else ""
        lines.append(
            f"{agent['agent']}：{agent['vote']}（信心{agent['confidence']}）｜{agent['reason']}{warning}"
        )

    lines.append("")
    lines.append(f"委员会投票：多头分 {result['bull_score']} / 空头分 {result['bear_score']}")
    lines.append(f"最终结论：{result['final_bias']}")
    lines.append(f"委员会信心：{result['confidence']} / 100")

    if result["warnings"]:
        lines.append("主要风险：" + " / ".join(result["warnings"]))

    lines.append("执行原则：如果委员会结论和风控脑冲突，以风控脑为优先。")
    return "\n".join(lines)


# =========================
# V32 AI Trading Brain
# Memory + Reflection + Self-Learning Layer
# =========================

def get_trading_brain_records(user_id="global"):
    data = load_json(TRADING_BRAIN_FILE, {})
    return data.get(str(user_id), [])


def save_trading_brain_records(user_id, records):
    data = load_json(TRADING_BRAIN_FILE, {})
    data[str(user_id)] = records[-TRADING_BRAIN_MAX_RECORDS:]
    save_json(TRADING_BRAIN_FILE, data)


def add_trading_brain_record(user_id, record):
    records = get_trading_brain_records(user_id)
    records.append(record)
    save_trading_brain_records(user_id, records)
    return record


def build_market_fingerprint(symbol, data, summary, news_risk_text=""):
    news_text = str(news_risk_text).lower()
    tags = []

    trend = data.get("trend", "震荡")
    structure = data.get("structure_event", "")
    risk = data.get("risk", "")
    atr_pct = safe_float(data.get("atr_pct", 0))
    rsi = safe_float(data.get("rsi", 50))

    tags.append(f"trend:{trend}")

    if "BOS 向上" in structure:
        tags.append("structure:bos_up")
    elif "BOS 向下" in structure:
        tags.append("structure:bos_down")
    elif "扫高" in structure:
        tags.append("structure:sweep_high")
    elif "扫低" in structure:
        tags.append("structure:sweep_low")
    else:
        tags.append("structure:neutral")

    if atr_pct >= 0.8:
        tags.append("volatility:high")
    elif atr_pct >= 0.35:
        tags.append("volatility:medium")
    else:
        tags.append("volatility:normal")

    if rsi >= 70:
        tags.append("rsi:overbought")
    elif rsi <= 30:
        tags.append("rsi:oversold")
    else:
        tags.append("rsi:neutral")

    if any(k in news_text for k in ["cpi", "非农", "fomc", "美联储", "利率", "pce", "ppi"]):
        tags.append("macro:high_impact")

    if any(k in news_text for k in ["待公布", "pending", "等待公布", "数据源暂未更新"]):
        tags.append("macro:pending")

    if any(k in news_text for k in ["已公布", "status:released"]):
        tags.append("macro:released")

    if "追多风险" in risk:
        tags.append("risk:avoid_chase_long")
    if "追空风险" in risk:
        tags.append("risk:avoid_chase_short")

    return tags


def find_similar_brain_records(user_id, fingerprint, symbol=None, limit=5):
    records = get_trading_brain_records(user_id)
    scored = []
    fp_set = set(fingerprint)

    for record in records:
        if symbol and record.get("symbol") != symbol:
            continue

        record_fp = set(record.get("fingerprint", []))
        if not record_fp:
            continue

        overlap = len(fp_set.intersection(record_fp))
        if overlap <= 0:
            continue

        score = overlap / max(len(fp_set), 1)
        scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def summarize_similar_records(records):
    if not records:
        return "暂无类似历史样本。"

    total = len(records)
    wins = sum(1 for r in records if r.get("outcome") == "win")
    losses = sum(1 for r in records if r.get("outcome") == "loss")
    pending = total - wins - losses

    lines = [
        f"类似历史样本：{total} 条",
        f"成功：{wins}，失败：{losses}，未复盘：{pending}",
    ]

    if wins + losses > 0:
        win_rate = round((wins / max(wins + losses, 1)) * 100, 1)
        lines.append(f"已复盘样本胜率：{win_rate}%")

    recent_notes = []
    for r in records[-3:]:
        note = r.get("reflection", "")
        if note:
            recent_notes.append(note[:80])

    if recent_notes:
        lines.append("近期复盘记忆：" + " / ".join(recent_notes))

    return "\n".join(lines)


def build_v32_brain_context(user_id, symbol, data, summary, news_risk_text=""):
    fingerprint = build_market_fingerprint(symbol, data, summary, news_risk_text)
    similar = find_similar_brain_records(user_id, fingerprint, symbol=symbol, limit=5)
    similar_text = summarize_similar_records(similar)

    return f"""
【V32交易大脑记忆层】
当前市场指纹：
{", ".join(fingerprint)}

历史类似情况：
{similar_text}

使用规则：
如果类似样本胜率偏低，要降低信心，不要给激进建议。
如果类似样本胜率偏高，也只能作为参考，仍必须结合当前关键位和风控。
""".strip()


def record_ai_decision(user_id, symbol, data, summary, decision_context, news_risk_text="", user_message=""):
    try:
        decision = detect_v30_trade_bias(symbol, data, summary, news_risk_text)
        fingerprint = build_market_fingerprint(symbol, data, summary, news_risk_text)

        record = {
            "id": f"{int(time.time())}_{symbol}",
            "time": format_local_time(),
            "symbol": symbol,
            "asset": get_asset_name(symbol),
            "user_message": user_message[:300],
            "price": data.get("price"),
            "support": data.get("support"),
            "resistance": data.get("resistance"),
            "trend": data.get("trend"),
            "structure": data.get("structure_event"),
            "bias": decision.get("bias"),
            "direction": decision.get("direction"),
            "confidence": decision.get("confidence"),
            "risk_level": decision.get("risk_level"),
            "fingerprint": fingerprint,
            "news_snapshot": str(news_risk_text)[:1200],
            "decision_context": str(decision_context)[:1200],
            "outcome": "pending",
            "reflection": "",
        }

        add_trading_brain_record(user_id, record)
        return record

    except Exception as e:
        print("V32 Record Decision Error:", e)
        return None


def review_pending_brain_records(user_id="global"):
    records = get_trading_brain_records(user_id)
    changed = False

    for record in records:
        if record.get("outcome") != "pending":
            continue

        try:
            symbol = record.get("symbol")
            direction = record.get("direction")
            old_price = safe_float(record.get("price"))

            record_time_id = str(record.get("id", "0")).split("_")[0]
            try:
                record_ts = int(record_time_id)
                if time.time() - record_ts < BRAIN_MIN_REVIEW_SECONDS:
                    continue
            except Exception:
                pass

            if not symbol or direction not in ["long", "short"] or old_price <= 0:
                continue

            current_price = get_realtime_price(symbol)
            if not current_price:
                continue

            move_pct = calculate_pct_move(current_price, old_price)

            if direction == "long":
                if move_pct >= 0.35:
                    record["outcome"] = "win"
                    record["reflection"] = f"多头判断后价格上涨约 {move_pct}%，方向有效。"
                elif move_pct <= -0.35:
                    record["outcome"] = "loss"
                    record["reflection"] = f"多头判断后价格下跌约 {move_pct}%，方向失败，需要复盘是否追高或宏观反向。"
            elif direction == "short":
                if move_pct <= -0.35:
                    record["outcome"] = "win"
                    record["reflection"] = f"空头判断后价格下跌约 {move_pct}%，方向有效。"
                elif move_pct >= 0.35:
                    record["outcome"] = "loss"
                    record["reflection"] = f"空头判断后价格上涨约 {move_pct}%，方向失败，需要复盘是否扫低反转或支撑未破。"

            if record.get("outcome") != "pending":
                record["reviewed_at"] = format_local_time()
                record["review_price"] = round_price(symbol, current_price)
                record["review_move_pct"] = move_pct
                changed = True

        except Exception as e:
            print("V32 Review Record Error:", e)

    if changed:
        save_trading_brain_records(user_id, records)

    return changed


async def check_v32_brain_reflection(context):
    try:
        review_pending_brain_records("global")
    except Exception as e:
        print("V32 Brain Reflection Error:", e)


def build_brain_summary(user_id="global"):
    records = get_trading_brain_records(user_id)

    if not records:
        return "交易大脑暂无记忆。你可以先让机器人多分析几次行情，系统会自动记录判断。"

    total = len(records)
    reviewed = [r for r in records if r.get("outcome") in ["win", "loss"]]
    wins = [r for r in reviewed if r.get("outcome") == "win"]
    losses = [r for r in reviewed if r.get("outcome") == "loss"]
    pending = total - len(reviewed)

    lines = ["【V32 交易大脑总结】"]
    lines.append(f"总记忆：{total} 条")
    lines.append(f"已复盘：{len(reviewed)} 条")
    lines.append(f"成功：{len(wins)}，失败：{len(losses)}，待复盘：{pending}")

    if reviewed:
        win_rate = round(len(wins) / max(len(reviewed), 1) * 100, 1)
        lines.append(f"历史判断胜率：{win_rate}%")

    bias_count = {}
    for r in records:
        bias = r.get("bias", "未知")
        bias_count[bias] = bias_count.get(bias, 0) + 1

    if bias_count:
        bias_text = " / ".join([f"{k}:{v}" for k, v in bias_count.items()])
        lines.append(f"方向分布：{bias_text}")

    recent_reflections = [r.get("reflection") for r in records[-10:] if r.get("reflection")]
    if recent_reflections:
        lines.append("最近复盘：")
        for item in recent_reflections[-3:]:
            lines.append(f"- {item}")

    lines.append("提醒：这是辅助学习记忆，不代表未来一定重复。")
    return "\n".join(lines)


# =========================
# V30 Trade Bias Engine
# Clear conclusion layer: bias / confidence / key levels / execution advice
# =========================

def clamp_value(value, low, high):
    return max(low, min(high, value))


def detect_v30_trade_bias(symbol, data, summary, news_risk_text=""):
    trend = data.get("trend", "震荡")
    avg_long = int(summary.get("avg_long", data.get("long_probability", 50)))
    avg_short = int(summary.get("avg_short", data.get("short_probability", 50)))
    structure = data.get("structure_event", "")
    risk = data.get("risk", "")
    news_text = str(news_risk_text).lower()

    macro_pending = any(k in news_text for k in ["待公布", "pending", "数据源暂未更新", "等待公布"])
    macro_released = any(k in news_text for k in ["status:released", "已公布"])
    high_news_risk = any(k in news_text for k in ["cpi", "非农", "fomc", "美联储", "利率", "pce", "ppi", "高影响"])

    bull = avg_long
    bear = avg_short

    if trend == "偏多":
        bull += 8
    elif trend == "偏空":
        bear += 8

    if "BOS 向上" in structure or "扫低" in structure:
        bull += 7
    if "BOS 向下" in structure or "扫高" in structure:
        bear += 7

    if "追多风险偏高" in risk:
        bull -= 8
    if "追空风险偏高" in risk:
        bear -= 8

    # Macro pending means less conviction, not automatic bullish/bearish.
    risk_penalty = 0
    if macro_pending and high_news_risk:
        risk_penalty += 12
    elif high_news_risk:
        risk_penalty += 6

    diff = bull - bear

    if diff >= 12:
        bias = "偏多"
        direction = "long"
    elif diff <= -12:
        bias = "偏空"
        direction = "short"
    else:
        bias = "震荡/观望"
        direction = "neutral"

    confidence = 50 + abs(diff)
    if trend != "震荡":
        confidence += 5
    if "BOS" in structure:
        confidence += 5
    confidence -= risk_penalty
    confidence = int(clamp_value(confidence, 25, 88))

    if confidence >= 72:
        confidence_label = "较高"
    elif confidence >= 58:
        confidence_label = "中等"
    else:
        confidence_label = "偏低"

    risk_level = "中等"
    if high_news_risk and macro_pending:
        risk_level = "高"
    elif "超买" in risk or "超卖" in risk or "追" in risk:
        risk_level = "偏高"
    elif confidence >= 70 and not high_news_risk:
        risk_level = "中低"

    return {
        "bias": bias,
        "direction": direction,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "risk_level": risk_level,
        "macro_pending": macro_pending,
        "macro_released": macro_released,
        "high_news_risk": high_news_risk,
    }



# =========================
# V31 Volatility Alert Engine
# =========================

def load_vol_alert_cache():
    return load_json(VOL_ALERT_CACHE_FILE, {})


def save_vol_alert_cache(data):
    save_json(VOL_ALERT_CACHE_FILE, data)


def should_send_vol_alert(symbol, direction):
    cache = load_vol_alert_cache()
    key = f"{symbol}_{direction}"

    last_time = cache.get(key, 0)
    now = time.time()

    if now - last_time < VOL_ALERT_COOLDOWN:
        return False

    cache[key] = now
    save_vol_alert_cache(cache)
    return True


def calculate_price_move(current_price, old_price):
    try:
        return round(current_price - old_price, 2)
    except Exception:
        return 0


def calculate_pct_move(current_price, old_price):
    try:
        return round(((current_price - old_price) / old_price) * 100, 2)
    except Exception:
        return 0


def detect_volatility_explosion(symbol, data):
    atr = safe_float(data.get("atr", 0))
    current_range = safe_float(data.get("current_candle_range", atr))

    if atr <= 0:
        return False

    return current_range >= atr * ATR_EXPLOSION_MULTIPLIER


def build_v31_reason(symbol, news_risk_text=""):
    reasons = []

    text = str(news_risk_text).lower()

    if any(k in text for k in ["cpi", "通胀"]):
        reasons.append("通胀数据影响")

    if any(k in text for k in ["fomc", "美联储", "利率"]):
        reasons.append("美联储/利率影响")

    if any(k in text for k in ["美元", "dxy"]):
        reasons.append("美元波动")

    if any(k in text for k in ["美债", "收益率"]):
        reasons.append("美债收益率变化")

    if any(k in text for k in ["战争", "中东", "避险"]):
        reasons.append("避险情绪变化")

    if not reasons:
        reasons.append("技术面资金推动")

    return "、".join(reasons[:3])


def build_v31_volatility_alert(symbol, timeframe, move_value, move_pct, direction, data, news_risk_text=""):
    asset = get_asset_name(symbol)

    support = data.get("support")
    resistance = data.get("resistance")
    price = data.get("price")

    reason = build_v31_reason(symbol, news_risk_text)

    if direction == "up":
        action = "急涨"
        trader_tip = "不要第一时间追多，等回踩确认更安全。"
    else:
        action = "急跌"
        trader_tip = "不要第一时间追空，等反抽确认更安全。"

    return f"""
【V31 突发行情提醒】

品种：{asset}
当前价格：{price}

检测到：{timeframe} {action}

波动：
价格变化：{move_value}
百分比变化：{move_pct}%

关键位：
支撑：{support}
压力：{resistance}

可能原因：
{reason}

交易员提醒：
{trader_tip}
当前市场可能进入高波动状态。
""".strip()


async def check_v31_volatility_alerts(context):
    try:
        symbols = ["XAUUSD", "BTCUSDT"]

        for symbol in symbols:
            try:
                multi_tf_data = analyze_multi_timeframe(symbol)

                if not multi_tf_data:
                    continue

                data = multi_tf_data.get("15m") or next(iter(multi_tf_data.values()))

                current_price = safe_float(data.get("price", 0))
                old_price = safe_float(data.get("prev_price", current_price))

                move_value = calculate_price_move(current_price, old_price)
                move_pct = calculate_pct_move(current_price, old_price)

                direction = "up" if move_value > 0 else "down"

                news_risk_text = build_news_risk_text(symbol)

                triggered = False
                timeframe = "1分钟"

                if symbol == "XAUUSD":
                    if abs(move_value) >= XAUUSD_1M_ALERT_USD:
                        triggered = True

                    if abs(move_value) >= XAUUSD_5M_ALERT_USD:
                        timeframe = "5分钟"

                elif symbol == "BTCUSDT":
                    if abs(move_pct) >= BTC_1M_ALERT_PCT:
                        triggered = True

                    if abs(move_pct) >= BTC_5M_ALERT_PCT:
                        timeframe = "5分钟"

                if detect_volatility_explosion(symbol, data):
                    triggered = True

                if not triggered:
                    continue

                if not should_send_vol_alert(symbol, direction):
                    continue

                alert_text = build_v31_volatility_alert(
                    symbol=symbol,
                    timeframe=timeframe,
                    move_value=move_value,
                    move_pct=move_pct,
                    direction=direction,
                    data=data,
                    news_risk_text=news_risk_text
                )

                try:
                    if TELEGRAM_CHAT_ID:
                        await context.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=alert_text + "\n\n以上仅供行情参考，不构成投资建议。"
                        )
                except Exception as send_error:
                    print("V31 Alert Send Error:", send_error)

            except Exception as symbol_error:
                print("V31 Symbol Error:", symbol_error)

    except Exception as e:
        print("V31 Volatility Engine Error:", e)


def build_v30_trade_decision_context(symbol, data, summary, news_risk_text=""):
    decision = detect_v30_trade_bias(symbol, data, summary, news_risk_text)
    asset = get_asset_name(symbol)

    support = data.get("support")
    resistance = data.get("resistance")
    price = data.get("price")
    long_low = data.get("entry_long_low")
    long_high = data.get("entry_long_high")
    short_low = data.get("entry_short_low")
    short_high = data.get("entry_short_high")
    stop_long = data.get("stop_loss_long")
    stop_short = data.get("stop_loss_short")

    direction = decision["direction"]

    if direction == "long":
        execute = f"不建议中间价硬追多；更适合等回踩 {long_low} ~ {long_high} 后出现止跌/重新站回短线均线，再考虑轻仓。"
        invalid = f"如果跌破 {stop_long}，多头思路先失效。"
        scenario_a = f"若站稳 {resistance} 上方，多头延续概率提高。"
        scenario_b = f"若回踩守住 {support}，更像健康回调。"
        scenario_c = f"若跌破 {support} 且反抽失败，短线转弱。"
    elif direction == "short":
        execute = f"不建议在急跌后追空；更适合等反弹 {short_low} ~ {short_high} 受压后，再考虑轻仓空。"
        invalid = f"如果突破 {stop_short}，空头思路先失效。"
        scenario_a = f"若跌破 {support}，空头延续概率提高。"
        scenario_b = f"若反弹不过 {resistance}，更像弱反抽。"
        scenario_c = f"若重新站稳 {resistance} 上方，空头思路失效。"
    else:
        execute = f"现在更偏观望；区间中间位置不要硬做，等靠近 {support} 或 {resistance} 后看确认。"
        invalid = f"真正站稳 {resistance} 上方或跌破 {support} 下方，才考虑顺势。"
        scenario_a = f"靠近 {support} 出现止跌，可观察短多反弹。"
        scenario_b = f"靠近 {resistance} 出现受压，可观察短空回落。"
        scenario_c = "如果上下都没有确认，继续观望比乱进更好。"

    if decision["high_news_risk"] and decision["macro_pending"]:
        news_rule = "当前有重要数据/消息风险且部分数据仍显示待公布，结论置信度要打折，不适合重仓。"
    elif decision["high_news_risk"]:
        news_rule = "当前存在重要宏观/新闻影响，价格可能反复扫，建议降低仓位。"
    else:
        news_rule = "当前主要按技术结构和关键位处理。"

    return f"""
【V30交易决策层】
品种：{asset}
当前价格：{price}
明确倾向：{decision['bias']}
AI信心：{decision['confidence']} / 100（{decision['confidence_label']}）
风险等级：{decision['risk_level']}

关键位：
支撑：{support}
压力：{resistance}

情景推演：
A. {scenario_a}
B. {scenario_b}
C. {scenario_c}

执行建议：
{execute}
{invalid}

风控提醒：
{news_rule}
""".strip()


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

    v22_context = build_v22_intelligence_context(symbol, data, summary, news_risk_text)
    v30_context = build_v30_trade_decision_context(symbol, data, summary, news_risk_text)
    v33_context = build_v33_committee_context(symbol, data, summary, news_risk_text)

    return f"""
当前价格：{price}
整体方向：{summary['overall']}
多周期结构：{summary['trend_text']}

{v22_context}

{v30_context}

{v33_context}

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
        "黄金": "XAUUSD",
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
            "price_source": data.get("price_source", "K线价"),
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
            f"{row['name']}：价格 {row['price']}（{row.get('price_source', 'K线价')}），趋势 {row['trend']}，RSI {row['rsi']}，"
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

{get_time_context()}

用户问：
{user_message}

市场总览资料：
{overview_text}

新闻/宏观风险：
{news_risk_text}

请像一个真人交易员，给市场状态判断。

要求：
- 第一行直接说：今天偏适合交易 / 不太适合重仓 / 偏观望。
- 不要逐个品种写报告。
- 只点 1~2 个最值得关注的品种。
- 说清楚主要风险来自哪里。
- 控制在 100~170 字。
- 不要喊单。
- 最后自然加：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.55
    )

    return response.choices[0].message.content

def compact_market_context(symbol, data, summary, news_risk_text, intent):
    asset_name = get_asset_name(symbol)

    v22_context = build_v22_intelligence_context(symbol, data, summary, news_risk_text)
    v30_context = build_v30_trade_decision_context(symbol, data, summary, news_risk_text)
    v32_context = build_v32_brain_context("global", symbol, data, summary, news_risk_text)
    v33_context = build_v33_committee_context(symbol, data, summary, news_risk_text)

    base = f"""
品种：{asset_name}
当前价格：{data['price']}（{data.get('price_source', 'K线价')}）
短线趋势：{data['trend']}
整体方向：{summary['overall']}
多周期结构：{summary['trend_text']}
支撑：{data['support']}
压力：{data['resistance']}
结构：{data['structure_event']}
风险：{data['risk']}

{v22_context}

{v30_context}

{v32_context}

{v33_context}

新闻/数据风险：
{news_risk_text}
"""

    if intent == "why_move":
        return base + f"""
重点只解释涨跌原因：
流动性：{data['liquidity']}
高低估：{data['premium_discount']}
"""

    if intent == "risk_check":
        return base + f"""
重点判断是否适合追/进：
回踩做多区：{data['entry_long_low']} ~ {data['entry_long_high']}
反弹做空区：{data['entry_short_low']} ~ {data['entry_short_high']}
"""

    if intent == "teaching":
        return f"""
用户正在问教学/概念。
只需要用简单话解释，不要强行给入场计划。
如果能结合当前 {asset_name} 行情，就简单举例。
当前结构：{data['structure_event']}
"""

    if intent == "macro_news":
        return f"""
用户重点问新闻/数据。
不要硬讲技术指标。
新闻/数据风险：
{news_risk_text}
品种：{asset_name}
当前趋势：{data['trend']}
"""

    return base + """
如果用户没有要求交易计划，不要输出计划A/计划B。
只给自然判断、关键风险、等待条件。
"""


def get_human_reply_rules(intent):
    if intent == "why_move":
        return "第一句直接说主要原因，然后补充 2~3 个因素。不要给计划A/B，不要硬给入场区。"
    if intent == "risk_check":
        return "第一句必须直接回答能不能追/能不能进，然后说明风险在哪里，最后告诉用户等什么确认。"
    if intent == "teaching":
        return "用普通人听得懂的话解释，不要像教科书，不要突然给交易计划。"
    if intent == "macro_news":
        return "重点解释新闻/数据对行情的影响。如果数据还没公布，就提醒公布前波动风险。"
    return "像真人交易员自然回复。先给结论，再讲原因，再讲怎么等。不要套模板。"

def generate_flexible_market_reply(user_message, symbol, multi_tf_data, summary, user_memory, news_risk_text):
    d15 = multi_tf_data["15m"]
    intent = detect_user_intent(user_message)
    asset_name = get_asset_name(symbol)
    asset_macro_note = get_asset_macro_note(symbol)
    learning_context = build_learning_context(str(user_memory.get("user_id", "")), symbol) if user_memory.get("user_id") else "暂无"

    context_text = compact_market_context(symbol, d15, summary, news_risk_text, intent)
    reply_rules = get_human_reply_rules(intent)

    prompt = f"""
{SYSTEM_PROMPT}

{get_time_context()}

用户原话：
{user_message}

识别到的意图：
{intent}

品种：
{asset_name}

品种风控重点：
{asset_macro_note}

用户学习记录/想法参考：
{learning_context if 'learning_context' in locals() else '暂无'}

只使用下面和用户问题有关的资料，不要全部硬塞进回复：
{context_text}

回复方式：
{reply_rules}

最重要：
- 第一行必须直接给明确结论：偏多 / 偏空 / 观望，不要含糊。
- 必须给：关键支撑、关键压力、什么情况继续、什么情况失效。
- 不要先讲指标。
- 不要像报告。
- 不要答非所问。
- 不要输出太多项目符号。
- 回复控制在 100~190 字，除非用户问教学。
- 只有用户明确问“怎么做/给计划/交易计划”，才可以用计划A/计划B。
- 语气要像真人交易员，不要像客服。
- 最后自然加：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.65
    )

    return response.choices[0].message.content

def generate_trade_plan_reply(user_message, symbol, multi_tf_data, summary, user_memory, news_risk_text):
    d15 = multi_tf_data["15m"]
    plan_text = build_trade_plan_text(symbol, d15, summary, news_risk_text)
    asset_name = get_asset_name(symbol)
    asset_macro_note = get_asset_macro_note(symbol)

    prompt = f"""
{SYSTEM_PROMPT}

{get_time_context()}

用户问：
{user_message}

品种：
{asset_name}

品种风控重点：
{asset_macro_note}

交易数据：
{plan_text}

这次用户明确问“怎么做/交易计划”，所以可以给计划A/计划B。
但不要像死板模板，要像真人交易员写给朋友看的交易计划。

请用这个格式：

【{asset_name} 交易计划】

当前看法：
先用 2 句话说清楚现在主方向和风险。

计划A：
方向：
入场区：
止损：
目标：
触发条件：
什么时候放弃：

计划B：
方向：
入场区：
止损：
目标：
触发条件：
什么时候放弃：

我的提醒：
用真人语气提醒仓位、数据风险、不要追单。

要求：
- 必须有计划A和计划B
- 但语言不要太机械
- 入场区、止损、目标要清楚
- 如果新闻数据风险大，要先提醒
- 不要叫重仓
- 不要说保证
- 最后写：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35
    )

    return response.choices[0].message.content

def generate_ai_reply(user_message, symbol, multi_tf_data, summary, user_memory, news_risk_text):
    d15 = multi_tf_data["15m"]
    asset_name = get_asset_name(symbol)

    prompt = f"""
{SYSTEM_PROMPT}

{get_time_context()}

用户问：
{user_message}

品种：{asset_name}

只参考这些核心信息：
当前价格：{d15['price']}（{d15.get('price_source', 'K线价')}）
短线趋势：{d15['trend']}
整体方向：{summary['overall']}
支撑：{d15['support']}
压力：{d15['resistance']}
结构：{d15['structure_event']}
风险：{d15['risk']}
新闻/数据：
{news_risk_text}

回复要求：
- 第一行直接给明确结论：偏多 / 偏空 / 观望
- 必须说清楚关键支撑、关键压力、失效条件
- 不要堆指标
- 不要像报告
- 不要强行给计划A/B
- 100~180 字
- 像真人交易员聊天
- 最后自然加：以上仅供行情参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )

    return response.choices[0].message.content

def build_chart_prompt(caption, user_memory, news_risk_text):
    return f"""
{VISION_PROMPT}

{get_time_context()}

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


def subscribe_alert(user_id, chat_id, symbol, direction="any"):
    alerts = load_json(ALERT_FILE, [])

    for item in alerts:
        if item["user_id"] == user_id and item["symbol"] == symbol and item.get("direction", "any") == direction:
            return False

    alerts.append({
        "user_id": user_id,
        "chat_id": chat_id,
        "symbol": symbol,
        "direction": direction,
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
昨天FOMC怎样？
昨天数据怎样？
最近一次CPI怎样？
CPI 会影响黄金吗？

V33 Multi-Agent Trading Committee Trader：
Full Macro Engine
- ForexFactory 经济日历抓取
- 实际值 / 市场预测 / 前值
- /today /tomorrow /nfp /cpi /jobless /fomc
- 中文快讯 + 宏观数据 + 技术面融合
- 突发新闻主动推送
- AI 自动交易计划A/B
- V22 市场状态识别 / 情景推演 / 置信度 / 宏观联动
- V25 Quiet ForexFactory + Circuit Breaker
- V26 GoldAPI 现货黄金 XAU/USD
- V27 Macro Event State：released/pending 防止误判已公布
- V28 Historical Macro：支持昨天/前天/最近一次/本周数据查询
- V29 修复：历史数据不会再被180分钟过滤器误删
- V30 Trade Bias Engine：明确方向、信心、关键位、失效条件、执行建议
- V31 Volatility Alert Engine：突发行情/暴涨暴跌/ATR异常提醒
- V32 AI Trading Brain：记忆/复盘/自学习/类似行情经验
- V33 Multi-Agent Committee：宏观脑/技术脑/流动性脑/情绪脑/风控脑/记忆脑投票
- 仓位计算
- 交易日志
- AI 复盘
- Macro Live：公布后自动刷新实际值
- 重要数据公布后主动推送
- /refreshmacro 强制刷新经济日历
/decision 查看交易决策层
/status 查看机器人状态
/macrostatus 查看宏观事件状态
- 想法库：记住你的交易想法
- 学习今天行情
- 总结你的交易风格
刷新经济日历 / 强制刷新

也可以：
订阅 BTC 提醒
如果黄金适合做多，提醒我
如果 BTC 跌破支撑，提醒我
仓位计算 账户1000u 风险2% 入场68000 止损67200
记录交易 BTC 多单 入场68000 止损67200 目标69500
我的交易日志
复盘我的交易
取消提醒
我的设置
/help
"""
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
支持功能：

/today 今日重要数据
/yesterday 昨日重要数据
/week 本周重要数据
/tomorrow 明日重要数据
/nfp 非农
/jobless 初请失业金
/cpi CPI
/fomc 美联储/FOMC
/refreshmacro 强制刷新经济日历
/decision 查看交易决策层
/status 查看机器人状态
/macrostatus 查看宏观事件状态
刷新经济日历 / 强制刷新

其他：
BTC 现在能买吗？
BTC 怎么做？
黄金怎么做？
EURUSD 怎么做？
黄金回踩哪里做多？
今天有什么重要数据？
明天非农怎么看？
昨天FOMC怎样？
昨天数据怎样？
最近一次CPI怎样？
这个图怎么看？直接发截图
订阅 BTC 提醒
如果黄金适合做多，提醒我
如果 BTC 跌破支撑，提醒我
仓位计算 账户1000u 风险2% 入场68000 止损67200
记录交易 BTC 多单 入场68000 止损67200 目标69500
我的交易日志
复盘我的交易
取消提醒
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








async def committee_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    memory = get_user_memory(user_id)
    symbol = memory.get("favorite_symbol", DEFAULT_SYMBOL)
    interval = memory.get("favorite_interval", DEFAULT_INTERVAL)

    try:
        news_risk_text = build_news_risk_text(symbol)
        multi_tf_data = analyze_multi_timeframe(symbol)
        multi_tf_data = ensure_multi_tf_data(symbol, interval, multi_tf_data)
        summary = build_multi_timeframe_summary(multi_tf_data)
        data = multi_tf_data.get("15m") or next(iter(multi_tf_data.values()))
        text = build_v33_committee_context(symbol, data, summary, news_risk_text)
        await update.message.reply_text(text + "\n\n以上仅供行情参考，不构成投资建议。")
    except Exception as e:
        print("Committee Command Error:", e)
        await update.message.reply_text("交易委员会暂时无法生成，请稍后再试。")


async def brain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(build_brain_summary("global"))


async def review_brain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    changed = review_pending_brain_records("global")
    if changed:
        await update.message.reply_text("已复盘部分历史判断，并更新交易大脑记忆。")
    else:
        await update.message.reply_text("暂时没有符合条件的新判断可以复盘。")


async def volatility_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "V31 波动监控已开启：\n"
        "- 黄金急涨急跌提醒\n"
        "- BTC暴涨暴跌提醒\n"
        "- ATR异常波动监控\n"
        "- 高波动状态提醒\n"
        "- 5分钟冷却防Spam"
    )


async def decision_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    memory = get_user_memory(user_id)
    symbol = memory.get("favorite_symbol", DEFAULT_SYMBOL)
    interval = memory.get("favorite_interval", DEFAULT_INTERVAL)

    try:
        news_risk_text = build_news_risk_text(symbol)
        multi_tf_data = analyze_multi_timeframe(symbol)
        multi_tf_data = ensure_multi_tf_data(symbol, interval, multi_tf_data)
        summary = build_multi_timeframe_summary(multi_tf_data)
        data = multi_tf_data.get("15m") or next(iter(multi_tf_data.values()))
        decision_text = build_v30_trade_decision_context(symbol, data, summary, news_risk_text)
        await update.message.reply_text(decision_text + "\n\n以上仅供行情参考，不构成投资建议。")
    except Exception as e:
        print("Decision Command Error:", e)
        await update.message.reply_text("暂时无法生成交易决策层，请稍后再试。")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = os.getenv("BOT_MODE", "polling")
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
    webhook_url = os.getenv("WEBHOOK_URL", "")

    text = f"""
【Bot 状态】

版本：V33 Multi-Agent Trading Committee + Spot Gold
运行模式：{mode}
Railway Domain：{railway_domain or '未检测到'}
Webhook URL：{webhook_url or '自动/未设置'}\nGoldAPI Key：{'已设置' if GOLDAPI_KEY else '未设置'}

核心功能：
- 行情分析
- V22 市场状态识别
- 情景推演
- AI 置信度
- 宏观联动
- V25 Quiet ForexFactory + Circuit Breaker
- V26 GoldAPI 现货黄金 XAU/USD
- V27 Macro Event State：released/pending 防止误判已公布
- V28 Historical Macro：支持昨天/前天/最近一次/本周数据查询
- V29 修复：历史数据不会再被180分钟过滤器误删
- V30 Trade Bias Engine：明确方向、信心、关键位、失效条件、执行建议
- V31 Volatility Alert Engine：突发行情/暴涨暴跌/ATR异常提醒
- V32 AI Trading Brain：记忆/复盘/自学习/类似行情经验
- V33 Multi-Agent Committee：宏观脑/技术脑/流动性脑/情绪脑/风控脑/记忆脑投票
- 中文快讯
- 突发新闻
- Macro Live
- 仓位计算
- 交易日志
- AI 复盘
- 自学习

如果宏观数据源 DNS 失败，V24 会自动重试并优先使用缓存，不会让机器人直接崩。
""".strip()

    await update.message.reply_text(text)



async def macro_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    memory = get_user_memory(user_id)
    symbol = memory.get("favorite_symbol", DEFAULT_SYMBOL)

    events = get_macro_events_live(days="today_tomorrow", symbol=symbol)
    if should_force_refresh_for_macro_question(events):
        events = filter_macro_events(days="today_tomorrow", symbol=symbol, force_refresh=True)

    await update.message.reply_text(build_macro_state_context(events))


async def refresh_macro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Clear circuit breaker so manual refresh really tries again.
        cache = load_json(MACRO_CACHE_FILE, {})
        cache["last_fail_at"] = 0
        cache["last_error"] = ""
        save_json(MACRO_CACHE_FILE, cache)

        fetch_forexfactory_calendar(days="today", force_refresh=True)
        fetch_forexfactory_calendar(days="today_tomorrow", force_refresh=True)
        fetch_forexfactory_calendar(days="yesterday", force_refresh=True)
        fetch_forexfactory_calendar(days="week", force_refresh=True)

        await update.message.reply_text("已强制刷新经济日历。你可以再问一次：昨天数据怎样？/ CPI 公布了吗？")
    except Exception as e:
        print("Refresh Macro Error:", e)
        await update.message.reply_text("刷新经济日历失败，可能是数据源暂时不可用。")



async def yesterday_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind=None, days="yesterday")


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind=None, days="week")


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind=None, days="today")


async def tomorrow_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind=None, days="tomorrow")


async def nfp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="nfp", days=detect_macro_time_range(user_message))


async def cpi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="cpi", days=detect_macro_time_range(user_message))


async def jobless_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="jobless", days=detect_macro_time_range(user_message))


async def fomc_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await macro_command(update, context, kind="fomc", days=detect_macro_time_range(user_message))



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

            direction = item.get("direction", "any")
            should_alert = should_send_direction_alert(summary, direction)
            reason = direction_alert_reason(summary, direction)

            if should_alert:
                asset_name = get_asset_name(symbol)
                message = f"""
行情提醒：{asset_name}

整体方向：{summary['overall']}
做多概率：{summary['avg_long']}%
做空概率：{summary['avg_short']}%

{reason}

别急着直接追，最好等关键位确认后再看。
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




# =========================
# V19 Position Sizing + Trade Journal
# =========================

def extract_number_after_keywords(text, keywords):
    text = text.lower().replace(",", "")

    for key in keywords:
        pattern = rf"{key}\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    return None


def extract_all_numbers(text):
    text = text.replace(",", "")
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    return [float(x) for x in nums]


def detect_position_side(text):
    lower = text.lower()

    if any(word in lower for word in ["做多", "多单", "买涨", "long"]):
        return "long"

    if any(word in lower for word in ["做空", "空单", "买跌", "short"]):
        return "short"

    return "unknown"


def is_position_size_request(user_message):
    text = user_message.lower()

    if any(word in text for word in POSITION_SIZE_KEYWORDS):
        if any(x in text for x in ["账户", "本金", "资金", "风险", "入场", "止损", "sl", "stop"]):
            return True

    return False


def is_trade_journal_request(user_message):
    text = user_message.lower()
    return any(word in text for word in TRADE_JOURNAL_KEYWORDS)


def parse_position_size_request(user_message):
    text = user_message.lower().replace(",", "")

    account = extract_number_after_keywords(text, ["账户", "本金", "资金", "account"])
    risk_pct = extract_number_after_keywords(text, ["风险", "risk"])
    entry = extract_number_after_keywords(text, ["入场", "进场", "entry"])
    stop = extract_number_after_keywords(text, ["止损", "sl", "stop"])

    nums = extract_all_numbers(text)

    if account is None and len(nums) >= 1:
        account = nums[0]

    if risk_pct is None:
        pct_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
        if pct_match:
            risk_pct = float(pct_match.group(1))

    if entry is None and len(nums) >= 3:
        entry = nums[-2]

    if stop is None and len(nums) >= 4:
        stop = nums[-1]
    elif stop is None and len(nums) >= 3:
        stop = nums[-1]

    return {
        "account": account,
        "risk_pct": risk_pct,
        "entry": entry,
        "stop": stop,
        "side": detect_position_side(text)
    }


def calculate_position_size(account, risk_pct, entry, stop):
    if not account or not risk_pct or not entry or not stop:
        return None

    risk_amount = account * risk_pct / 100
    stop_distance = abs(entry - stop)

    if stop_distance <= 0:
        return None

    qty = risk_amount / stop_distance
    notional = qty * entry

    return {
        "risk_amount": risk_amount,
        "stop_distance": stop_distance,
        "qty": qty,
        "notional": notional
    }


def build_position_size_reply(user_message, symbol):
    data = parse_position_size_request(user_message)
    result = calculate_position_size(
        data.get("account"),
        data.get("risk_pct"),
        data.get("entry"),
        data.get("stop")
    )

    asset_name = get_asset_name(symbol)

    if not result:
        return """
你这句资料还不够完整。

你可以这样发：
仓位计算 账户1000u 风险2% 入场68000 止损67200

我会帮你算最大亏损、建议数量和大概名义仓位。

以上仅供行情参考，不构成投资建议。
""".strip()

    return f"""
【{asset_name} 仓位计算】

账户资金：{data['account']}
单笔风险：{data['risk_pct']}%
最大可亏：{round(result['risk_amount'], 2)}

入场价：{data['entry']}
止损价：{data['stop']}
止损距离：{round(result['stop_distance'], 4)}

建议数量：{round(result['qty'], 6)}
名义仓位约：{round(result['notional'], 2)}

我的建议：
这个算法是按“打到止损只亏账户 {data['risk_pct']}%”来算的。
如果今晚有数据或行情波动很大，可以把风险降到 0.5%~1%。

以上仅供行情参考，不构成投资建议。
""".strip()


def parse_trade_record(user_message, symbol):
    text = user_message.lower().replace(",", "")

    side = detect_position_side(text)

    entry = extract_number_after_keywords(text, ["入场", "进场", "entry"])
    stop = extract_number_after_keywords(text, ["止损", "sl", "stop"])
    target = extract_number_after_keywords(text, ["目标", "tp", "target", "止盈"])

    nums = extract_all_numbers(text)

    if entry is None and len(nums) >= 1:
        entry = nums[0]

    if stop is None and len(nums) >= 2:
        stop = nums[1]

    if target is None and len(nums) >= 3:
        target = nums[2]

    return {
        "symbol": symbol,
        "asset_name": get_asset_name(symbol),
        "side": side,
        "entry": entry,
        "stop": stop,
        "target": target,
        "note": user_message,
        "created_at": get_local_now().strftime("%Y-%m-%d %H:%M UTC+8"),
        "status": "open"
    }


def add_trade_record(user_id, record):
    journal = load_json(TRADE_JOURNAL_FILE, {})

    if user_id not in journal:
        journal[user_id] = []

    journal[user_id].append(record)
    journal[user_id] = journal[user_id][-100:]

    save_json(TRADE_JOURNAL_FILE, journal)


def get_user_trades(user_id):
    journal = load_json(TRADE_JOURNAL_FILE, {})
    return journal.get(user_id, [])


def build_trade_record_reply(user_id, user_message, symbol):
    record = parse_trade_record(user_message, symbol)

    if not record["entry"]:
        return """
你想记录交易的话，可以这样发：

记录交易 BTC 多单 入场68000 止损67200 目标69500

之后可以输入：
我的交易日志
复盘我的交易
""".strip()

    add_trade_record(user_id, record)

    side_text = "多单" if record["side"] == "long" else ("空单" if record["side"] == "short" else "方向未注明")

    return f"""
已帮你记录这笔交易。

品种：{record['asset_name']}
方向：{side_text}
入场：{record['entry']}
止损：{record['stop'] or '未填写'}
目标：{record['target'] or '未填写'}
时间：{record['created_at']}

之后你可以发：
我的交易日志
复盘我的交易
""".strip()


def build_trade_journal_reply(user_id):
    trades = get_user_trades(user_id)

    if not trades:
        return "你目前还没有交易记录。可以发：记录交易 BTC 多单 入场68000 止损67200 目标69500"

    recent = trades[-10:]
    lines = []

    for i, trade in enumerate(recent, start=1):
        side_text = "多单" if trade.get("side") == "long" else ("空单" if trade.get("side") == "short" else "未注明")
        lines.append(
            f"{i}. {trade.get('asset_name')}｜{side_text}｜入场 {trade.get('entry')}｜止损 {trade.get('stop') or '未填'}｜目标 {trade.get('target') or '未填'}｜{trade.get('created_at')}"
        )

    return "【最近交易日志】\n\n" + "\n".join(lines)


def build_trade_review_reply(user_id):
    trades = get_user_trades(user_id)

    if not trades:
        return "你目前还没有交易记录，暂时没法复盘。"

    recent = trades[-20:]
    text = json.dumps(recent, ensure_ascii=False, indent=2)

    prompt = f"""
你是一个交易复盘教练。
请根据用户最近交易记录，帮他做复盘。

交易记录：
{text}

要求：
- 不要编造成交结果
- 如果没有平仓结果，就只分析计划质量
- 找出常见问题：追单、止损太近、止损太远、方向不清、没有目标、没写理由
- 给 3 条改进建议
- 语气像真人教练，不要太严厉
- 最后提醒：以上仅供复盘参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.45
    )

    return response.choices[0].message.content



# =========================
# V21 Self Learning Trader
# =========================

def is_self_learning_request(user_message):
    text = user_message.lower()
    return any(keyword.lower() in text for keyword in SELF_LEARNING_KEYWORDS)


def extract_after_prefix(message, prefixes):
    text = message.strip()
    for prefix in prefixes:
        if prefix in text:
            return text.split(prefix, 1)[1].strip(" ：:")
    return ""


def get_user_ideas(user_id):
    data = load_json(USER_IDEA_FILE, {})
    return data.get(user_id, [])


def save_user_ideas(user_id, ideas):
    data = load_json(USER_IDEA_FILE, {})
    data[user_id] = ideas[-100:]
    save_json(USER_IDEA_FILE, data)


def add_user_idea(user_id, idea_text, symbol=None):
    ideas = get_user_ideas(user_id)
    item = {
        "time": format_local_time() if "format_local_time" in globals() else datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol or "",
        "idea": idea_text
    }
    ideas.append(item)
    save_user_ideas(user_id, ideas)
    return item


def delete_user_ideas(user_id):
    save_user_ideas(user_id, [])


def build_user_ideas_text(user_id, limit=8):
    ideas = get_user_ideas(user_id)
    if not ideas:
        return "你目前还没有记录交易想法。可以发：记住我的想法：CPI低于预期时，我偏向先看黄金多头。"

    recent = ideas[-limit:]
    lines = []
    for i, item in enumerate(recent, start=1):
        symbol_text = f"｜{get_asset_name(item.get('symbol'))}" if item.get("symbol") else ""
        lines.append(f"{i}. {item.get('time')}{symbol_text}\n{item.get('idea')}")
    return "【我的想法库】\n\n" + "\n\n".join(lines)


def get_learning_logs(user_id):
    data = load_json(LEARNING_LOG_FILE, {})
    return data.get(user_id, [])


def save_learning_logs(user_id, logs):
    data = load_json(LEARNING_LOG_FILE, {})
    data[user_id] = logs[-100:]
    save_json(LEARNING_LOG_FILE, data)


def add_learning_log(user_id, title, content, symbol=None):
    logs = get_learning_logs(user_id)
    item = {
        "time": format_local_time() if "format_local_time" in globals() else datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": title,
        "symbol": symbol or "",
        "content": content
    }
    logs.append(item)
    save_learning_logs(user_id, logs)
    return item


def get_market_thoughts(user_id):
    data = load_json(MARKET_THOUGHT_FILE, {})
    return data.get(user_id, [])


def save_market_thoughts(user_id, thoughts):
    data = load_json(MARKET_THOUGHT_FILE, {})
    data[user_id] = thoughts[-100:]
    save_json(MARKET_THOUGHT_FILE, data)


def add_market_thought(user_id, content, symbol=None):
    thoughts = get_market_thoughts(user_id)
    item = {
        "time": format_local_time() if "format_local_time" in globals() else datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol or "",
        "content": content
    }
    thoughts.append(item)
    save_market_thoughts(user_id, thoughts)
    return item


def build_learning_context(user_id, symbol=None):
    ideas = get_user_ideas(user_id)[-5:]
    logs = get_learning_logs(user_id)[-5:]
    thoughts = get_market_thoughts(user_id)[-5:]
    parts = []

    idea_lines = []
    for item in ideas:
        if symbol and item.get("symbol") and item.get("symbol") != symbol:
            continue
        idea_lines.append(f"- {item.get('idea')}")
    if idea_lines:
        parts.append("用户交易想法：\n" + "\n".join(idea_lines))

    log_lines = []
    for item in logs:
        log_lines.append(f"- {item.get('title')}: {item.get('content', '')[:160]}")
    if log_lines:
        parts.append("历史学习记录：\n" + "\n".join(log_lines))

    thought_lines = []
    for item in thoughts:
        if symbol and item.get("symbol") and item.get("symbol") != symbol:
            continue
        thought_lines.append(f"- {item.get('content', '')[:160]}")
    if thought_lines:
        parts.append("市场想法记录：\n" + "\n".join(thought_lines))

    if not parts:
        return "暂无用户想法或学习记录。"
    return "\n\n".join(parts)


def build_add_idea_reply(user_id, user_message, symbol):
    idea = extract_after_prefix(
        user_message,
        ["记住我的想法", "记录我的想法", "我的想法是", "我的交易想法", "记录市场想法", "市场想法"]
    )
    if not idea:
        return "可以，你这样发：记住我的想法：CPI低于预期时，我偏向先看黄金多头。"

    item = add_user_idea(user_id, idea, symbol)
    return f"""
已记住你的想法。

品种：{get_asset_name(symbol)}
时间：{item['time']}
想法：{idea}

以后你问相关行情时，我会把这个想法作为参考之一。
""".strip()


def build_market_learning_reply(user_id, user_message, symbol):
    try:
        user_memory = get_user_memory(user_id)
        news_risk_text = build_news_risk_text(symbol)
        multi_tf_data = analyze_multi_timeframe(symbol)
        multi_tf_data = ensure_multi_tf_data(symbol, DEFAULT_INTERVAL, multi_tf_data)
        summary = build_multi_timeframe_summary(multi_tf_data)
        d15 = multi_tf_data["15m"]
        ideas_text = build_learning_context(user_id, symbol)

        prompt = f"""
{SYSTEM_PROMPT}

{get_time_context() if "get_time_context" in globals() else ""}

用户要求：
{user_message}

品种：
{get_asset_name(symbol)}

当前市场资料：
价格：{d15['price']}（{d15.get('price_source', 'K线价')}）
短线趋势：{d15['trend']}
整体方向：{summary['overall']}
多周期结构：{summary['trend_text']}
支撑：{d15['support']}
压力：{d15['resistance']}
结构：{d15['structure_event']}
风险：{d15['risk']}

新闻/数据：
{news_risk_text}

用户想法/学习记录：
{ideas_text}

请做一份“今天行情学习总结”。

要求：
- 不要喊单
- 重点总结今天行情给我的经验
- 说明哪里容易误判
- 给 3 条下次改进重点
- 语气像交易教练
- 最后写：以上仅供复盘参考，不构成投资建议。
"""

        response = client.chat.completions.create(
            model=TEXT_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.45
        )
        content = response.choices[0].message.content
        add_learning_log(user_id, "行情学习总结", content, symbol)
        return content

    except Exception as e:
        print("Market Learning Error:", e)
        return "今天行情学习总结暂时生成失败，可能是行情或新闻数据源暂时不稳定。"


def build_style_summary_reply(user_id):
    trades = get_user_trades(user_id) if "get_user_trades" in globals() else []
    ideas = get_user_ideas(user_id)
    logs = get_learning_logs(user_id)
    thoughts = get_market_thoughts(user_id)

    if not trades and not ideas and not logs and not thoughts:
        return "目前资料还不够总结你的交易风格。你可以先记录几笔交易，或发：记住我的想法：xxx"

    prompt = f"""
你是一个交易教练。
请根据用户的交易记录、想法库和学习日志，总结他的交易风格。

交易记录：
{json.dumps(trades[-20:], ensure_ascii=False, indent=2)}

想法库：
{json.dumps(ideas[-20:], ensure_ascii=False, indent=2)}

学习日志：
{json.dumps(logs[-10:], ensure_ascii=False, indent=2)}

市场想法：
{json.dumps(thoughts[-20:], ensure_ascii=False, indent=2)}

请输出：
1. 交易风格画像
2. 优势
3. 可能的弱点
4. 最需要改进的 3 件事
5. 适合他的风控建议

要求：
- 不要编造成交结果
- 如果资料不足，要说明只是初步判断
- 语气像真人教练
- 最后写：以上仅供复盘参考，不构成投资建议。
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.45
    )
    return response.choices[0].message.content


def handle_self_learning_text(user_id, user_message, symbol):
    if "清空想法" in user_message or "删除想法" in user_message:
        delete_user_ideas(user_id)
        return "已清空你的想法库。"

    if "我的想法库" in user_message:
        return build_user_ideas_text(user_id)

    if any(key in user_message for key in ["记住我的想法", "记录我的想法", "我的想法是", "我的交易想法", "记录市场想法", "市场想法"]):
        return build_add_idea_reply(user_id, user_message, symbol)

    if any(key in user_message for key in ["学习今天行情", "总结今天行情", "今天复盘"]):
        return build_market_learning_reply(user_id, user_message, symbol)

    if any(key in user_message for key in ["总结我的交易风格", "我的交易风格", "学习我的风格"]):
        return build_style_summary_reply(user_id)

    return "你可以说：记住我的想法：xxx，或者：学习今天行情 / 我的想法库 / 总结我的交易风格。"



def detect_alert_direction(user_message):
    text = user_message.lower()

    if any(word in text for word in ["做多", "买涨", "上涨", "上去", "突破", "适合多", "适合做多", "可以多", "多单"]):
        return "long"

    if any(word in text for word in ["做空", "买跌", "下跌", "跌破", "适合空", "适合做空", "可以空", "空单"]):
        return "short"

    return "any"


def is_alert_request(user_message):
    text = user_message.lower()

    if "提醒" not in text and "通知" not in text and "叫我" not in text:
        return False

    alert_words = [
        "如果", "当", "一旦", "适合", "可以", "上涨", "下跌",
        "突破", "跌破", "做多", "做空", "买涨", "买跌",
        "入场", "进场", "机会"
    ]

    return any(word in text for word in alert_words)


def build_alert_reply(symbol, direction):
    asset_name = get_asset_name(symbol)

    if direction == "long":
        return f"可以，我帮你盯着 {asset_name}。如果出现偏适合做多/上涨延续的条件，我会提醒你。"
    if direction == "short":
        return f"可以，我帮你盯着 {asset_name}。如果出现偏适合做空/下跌延续的条件，我会提醒你。"

    return f"可以，我帮你盯着 {asset_name}。如果行情出现明显机会或风险变化，我会提醒你。"


def should_send_direction_alert(summary, direction):
    if direction == "long":
        return summary["avg_long"] >= 68

    if direction == "short":
        return summary["avg_short"] >= 68

    return summary["avg_long"] >= 75 or summary["avg_short"] >= 75


def direction_alert_reason(summary, direction):
    if direction == "long":
        return f"做多条件开始转强，当前做多概率约 {summary['avg_long']}%。"

    if direction == "short":
        return f"做空条件开始转强，当前做空概率约 {summary['avg_short']}%。"

    if summary["avg_long"] >= summary["avg_short"]:
        return f"行情开始偏多，当前做多概率约 {summary['avg_long']}%。"

    return f"行情开始偏空，当前做空概率约 {summary['avg_short']}%。"



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

    # =========================
    # V21 Self Learning Trader
    # =========================
    if is_self_learning_request(user_message):
        reply = handle_self_learning_text(user_id, user_message, symbol)
        await update.message.reply_text(reply)
        return

    # =========================
    # V20.1 Macro Refresh Text Fallback
    # 群组里 /refreshmacro 有时不会触发时，用普通文字也可以刷新
    # =========================
    if (
        user_message.startswith("/refreshmacro")
        or "刷新经济日历" in user_message
        or "强制刷新" in user_message
        or "刷新macro" in user_message.lower()
        or "刷新宏观" in user_message
    ):
        try:
            fetch_forexfactory_calendar(days="today", force_refresh=True)
            fetch_forexfactory_calendar(days="today_tomorrow", force_refresh=True)
            await update.message.reply_text("已强制刷新经济日历。你可以再问一次：CPI 公布了吗？")
        except Exception as e:
            print("Refresh Macro Text Error:", e)
            await update.message.reply_text("刷新经济日历失败，可能是数据源暂时不可用。")
        return

    # =========================
    # 取消提醒 / 关闭推送
    # =========================
    if (
        "取消提醒" in user_message
        or "停止提醒" in user_message
        or "关闭提醒" in user_message
        or "不要提醒了" in user_message
        or "关闭行情提醒" in user_message
        or "取消行情提醒" in user_message
        or "取消突发行情" in user_message
        or "关闭突发行情" in user_message
        or "停止突发行情" in user_message
    ):
        alerts = load_json(ALERT_FILE, [])

        old_count = len(alerts)

        alerts = [
            item for item in alerts
            if str(item.get("user_id")) != str(user_id)
        ]

        removed_count = old_count - len(alerts)

        save_json(ALERT_FILE, alerts)

        if removed_count > 0:
            await update.message.reply_text(
                "已帮你关闭所有行情提醒和突发新闻推送。"
            )
        else:
            await update.message.reply_text(
                "你目前没有开启中的行情提醒。"
            )

        return

    if is_position_size_request(user_message):
        reply = build_position_size_reply(user_message, symbol)
        await update.message.reply_text(reply)
        return

    if "我的交易日志" in user_message or user_message == "交易日志":
        reply = build_trade_journal_reply(user_id)
        await update.message.reply_text(reply)
        return

    if "复盘我的交易" in user_message or user_message == "复盘":
        reply = build_trade_review_reply(user_id)
        await update.message.reply_text(reply)
        return

    if is_trade_journal_request(user_message):
        reply = build_trade_record_reply(user_id, user_message, symbol)
        await update.message.reply_text(reply)
        return

    if is_alert_request(user_message):
        direction = detect_alert_direction(user_message)
        ok = subscribe_alert(user_id, chat_id, symbol, direction)

        if ok:
            await update.message.reply_text(build_alert_reply(symbol, direction))
        else:
            await update.message.reply_text("这个提醒我已经帮你设置过了。")

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

    if any(k in user_message for k in ["昨天", "昨日", "前天", "上次", "最近一次", "上一份"]):
        if any(k in lower for k in ["数据", "新闻", "事件", "fomc", "cpi", "nfp", "jobless", "pce", "ppi", "gdp", "retail", "pmi"]) or any(k in user_message for k in ["非农", "初请", "失业金", "美联储", "利率决议", "通胀", "零售"]):
            report = build_macro_report_from_message(user_message, symbol=symbol)
            await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
            return

    if "今天" in user_message and ("数据" in user_message or "新闻" in user_message or "事件" in user_message):
        report = build_macro_report_from_message(user_message, symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "明天" in user_message and ("数据" in user_message or "新闻" in user_message or "事件" in user_message or "非农" in user_message):
        report = build_macro_report_from_message(user_message, symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "非农" in user_message or "nfp" in lower:
        report = build_macro_report(kind="nfp", days=detect_macro_time_range(user_message), symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "初请" in user_message or "失业金" in user_message or "jobless" in lower:
        report = build_macro_report(kind="jobless", days=detect_macro_time_range(user_message), symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "cpi" in lower or "通胀" in user_message:
        report = build_macro_report(kind="cpi", days=detect_macro_time_range(user_message), symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "fomc" in lower or "美联储" in user_message or "利率决议" in user_message:
        report = build_macro_report(kind="fomc", days=detect_macro_time_range(user_message), symbol=symbol)
        await update.message.reply_text(f"{report}\n\n以上仅供行情参考，不构成投资建议。")
        return

    if "订阅" in user_message and "提醒" in user_message:
        ok = subscribe_alert(user_id, chat_id, symbol, detect_alert_direction(user_message))

        direction = detect_alert_direction(user_message)

        if ok:
            await update.message.reply_text(build_alert_reply(symbol, direction))
        else:
            await update.message.reply_text("这个提醒我已经帮你设置过了。")

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

    try:
        if "multi_tf_data" in locals() and "summary" in locals() and "symbol" in locals() and "news_risk_text" in locals():
            record_data = None
            if isinstance(multi_tf_data, dict):
                record_data = multi_tf_data.get("15m") or next(iter(multi_tf_data.values()), None)
            if record_data:
                decision_context = build_v30_trade_decision_context(symbol, record_data, summary, news_risk_text)
                record_ai_decision("global", symbol, record_data, summary, decision_context, news_risk_text, user_message)
    except Exception as brain_error:
        print("V32 Brain Record Hook Error:", brain_error)

    await update.message.reply_text(reply)


def main():
    validate_env()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("yesterday", yesterday_command))
    app.add_handler(CommandHandler("week", week_command))
    app.add_handler(CommandHandler("tomorrow", tomorrow_command))
    app.add_handler(CommandHandler("nfp", nfp_command))
    app.add_handler(CommandHandler("cpi", cpi_command))
    app.add_handler(CommandHandler("jobless", jobless_command))
    app.add_handler(CommandHandler("fomc", fomc_command))
    app.add_handler(CommandHandler("refreshmacro", refresh_macro_command))
    app.add_handler(CommandHandler("committee", committee_command))
    app.add_handler(CommandHandler("brain", brain_command))
    app.add_handler(CommandHandler("reviewbrain", review_brain_command))
    app.add_handler(CommandHandler("volatility", volatility_command))
    app.add_handler(CommandHandler("decision", decision_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("macrostatus", macro_status_command))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.job_queue.run_repeating(check_alerts, interval=300, first=30)
    app.job_queue.run_repeating(check_breaking_news, interval=180, first=45)
    app.job_queue.run_repeating(check_v32_brain_reflection, interval=1800, first=600)
    app.job_queue.run_repeating(check_v31_volatility_alerts, interval=60, first=30)
    app.job_queue.run_repeating(check_macro_live_releases, interval=600, first=120)

    print("V33 Multi-Agent Trading Committee Trader 已启动...")
    print("API：OpenRouter")
    print("文字模型：", TEXT_MODEL_NAME)
    print("图片模型：", VISION_MODEL_NAME)
    print("宏观引擎：ForexFactory + 中文快讯")
    print("本地时间：", format_local_time())
    print("已开启：V30 Trade Bias Engine + V29 Historical Macro Filter Fix + V28 Historical Macro + V27 Macro Event State + V26 GoldAPI Spot Gold + V25 Quiet ForexFactory + Circuit Breaker + V22市场状态识别 + 情景推演 + 置信度 + 宏观联动 + Self Learning + Macro Live + 仓位计算 + AI复盘 + 条件提醒")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
