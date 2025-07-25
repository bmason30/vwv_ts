"""
Constants and static data for the VWV Trading System
"""

# Symbol descriptions for quick links
SYMBOL_DESCRIPTIONS = {
    # Index ETFs
    'SPY': 'SPDR S&P 500 ETF - Large Cap US Stocks',
    'VOO': 'Vanguard S&P 500 ETF - Low Cost S&P 500',
    'QQQ': 'Invesco QQQ Trust - Nasdaq-100 ETF',
    'IWM': 'iShares Russell 2000 ETF - Small Cap US Stocks',
    'MAGS': 'Roundhill Magnificent Seven ETF',
    'SPHB': 'Invesco S&P 500 High Beta ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    
    # International
    'EWW': 'iShares MSCI Mexico ETF',
    'FXI': 'iShares China Large-Cap ETF',
    'INDA': 'iShares MSCI India ETF',
    'UUP': 'Invesco DB US Dollar Bullish ETF',
    'UDN': 'Invesco DB US Dollar Bearish ETF',
    
    # Commodities
    'GLD': 'SPDR Gold Shares - Physical Gold ETF',
    'GDX': 'VanEck Gold Miners ETF - Gold Mining Stocks',
    'SLV': 'iShares Silver Trust - Physical Silver ETF',
    'URNM': 'North Shore Global Uranium Mining ETF',
    'PHYS': 'Sprott Physical Gold Trust',
    
    # Income ETFs
    'JEPI': 'JPMorgan Equity Premium Income ETF',
    'DIVO': 'Amplify CWP Enhanced Dividend Income ETF',
    'SCHD': 'Schwab US Dividend Equity ETF',
    'SPYI': 'NEOS S&P 500 High Income ETF',
    'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
    'JNK': 'SPDR Bloomberg High Yield Bond ETF',
    
    # Tech Giants
    'TSLA': 'Tesla Inc - Electric Vehicles & Clean Energy',
    'AAPL': 'Apple Inc - Consumer Electronics & Technology',
    'MSFT': 'Microsoft Corporation - Software & Cloud Services',
    'NVDA': 'NVIDIA Corporation - Graphics & AI Chips',
    'AMZN': 'Amazon.com Inc - E-commerce & Cloud Computing',
    'GOOGL': 'Alphabet Inc - Google Search & Cloud',
    'NFLX': 'Netflix Inc - Streaming Entertainment',
    'META': 'Meta Platforms Inc - Social Media & Metaverse',
    
    # Semiconductors
    'CHIPS': 'SPDR S&P Semiconductor ETF',
    'SMCI': 'Super Micro Computer Inc - AI Server Hardware',
    'INTC': 'Intel Corporation - Semiconductor Chips',
    'MU': 'Micron Technology - Memory & Storage',
    'AVGO': 'Broadcom Inc - Semiconductor Solutions',
    'AMD': 'Advanced Micro Devices - CPU & GPU',
    'LRCX': 'Lam Research - Semiconductor Equipment',
    'QCOM': 'Qualcomm Inc - Mobile Chip Technology',
    'SOXL': 'Direxion Semiconductor Bull 3X ETF',
    
    # Software & AI
    'NET': 'Cloudflare Inc - Web Infrastructure & Security',
    'PLTR': 'Palantir Technologies - Big Data Analytics',
    'SNOW': 'Snowflake Inc - Cloud Data Platform',
    'PANW': 'Palo Alto Networks - Cybersecurity',
    'ORCL': 'Oracle Corporation - Database Software',
    'AI': 'C3.ai Inc - Enterprise AI Software',
    
    # Blue Chips
    'UNH': 'UnitedHealth Group - Healthcare & Insurance',
    'HD': 'The Home Depot - Home Improvement Retail',
    'COST': 'Costco Wholesale - Membership Retail',
    'WMT': 'Walmart Inc - Retail & E-commerce',
    'V': 'Visa Inc - Payment Processing',
    'GS': 'Goldman Sachs Group - Investment Banking',
    'DIS': 'The Walt Disney Company - Entertainment',
    'CAT': 'Caterpillar Inc - Heavy Machinery',
    'BA': 'Boeing Company - Aerospace & Defense',
    'XOM': 'Exxon Mobil Corporation - Oil & Gas',
    
    # Leveraged ETFs
    'FNGD': 'MicroSectors FANG+ 3X Inverse Leveraged ETN',
    'FNGU': 'MicroSectors FANG+ 3X Leveraged ETN',
    'TZA': 'Direxion Small Cap Bear 3X ETF',
    
    # Crypto & Digital Assets
    'FETH': 'Fidelity Ethereum ETF - Crypto Exposure',
    'BTC': 'Bitcoin ETF - Digital Asset Exposure',
    'IBIT': 'iShares Bitcoin Trust ETF',
    'COIN': 'Coinbase Global Inc - Crypto Exchange',
    'MARA': 'Marathon Digital Holdings - Bitcoin Mining',
    
    # Others
    'AIG': 'American International Group - Insurance',
    'GOOG': 'Alphabet Inc Class C - Google Parent'
}

# Quick links categories
QUICK_LINK_CATEGORIES = {
    '📈 Index ETFs': ['SPY', 'VOO', 'QQQ', 'IWM', 'MAGS', 'SPHB', 'TLT'],
    '🌍 International': ['EWW', 'FXI', 'INDA', 'UUP', 'UDN'],
    '🥇 Commodities': ['GLD', 'GDX', 'SLV', 'URNM', 'PHYS'],
    '💰 Income ETFs': ['JEPI', 'DIVO', 'SCHD', 'SPYI', 'HYG', 'JNK'],
    '🚀 Tech Giants': ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'NFLX', 'META'],
    '💾 Semiconductors': ['CHIPS', 'SMCI', 'INTC', 'MU', 'AVGO', 'AMD', 'LRCX', 'QCOM', 'SOXL'],
    '🌐 Software & AI': ['NET', 'PLTR', 'SNOW', 'PANW', 'ORCL', 'AI'],
    '🏢 Blue Chips': ['UNH', 'HD', 'COST', 'WMT', 'V', 'GS', 'DIS', 'CAT', 'BA', 'XOM'],
    '⚡ Leveraged': ['FNGD', 'FNGU', 'TZA'],
    '🪙 Crypto & Digital': ['FETH', 'BTC', 'IBIT', 'COIN', 'MARA'],
    '📺 Other Stocks': ['AIG', 'GOOG']
}

# Known individual stocks (not ETFs)
KNOWN_INDIVIDUAL_STOCKS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE',
    'KO', 'ADBE', 'PEP', 'TMO', 'COST', 'AVGO', 'NKE', 'MRK', 'ABT', 'CRM',
    'LLY', 'ACN', 'TXN', 'DHR', 'WMT', 'NEE', 'VZ', 'ORCL', 'CMCSA', 'PM',
    'DIS', 'BMY', 'RTX', 'HON', 'QCOM', 'UPS', 'T', 'AIG', 'LOW', 'MDT'
}

# Known ETFs
KNOWN_ETFS = {
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
    'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
    'XLY', 'XLU', 'XLRE', 'XLB', 'EFA', 'EEM', 'FXI', 'EWJ', 'EWG', 'EWU',
    'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'FNGU', 'FNGD', 'MAGS', 'SOXX',
    'SMH', 'IBB', 'XBI', 'JETS', 'HACK', 'ESPO', 'ICLN', 'PBW', 'KWEB',
    'SPHB', 'SOXL', 'QQI', 'DIVO', 'URNM', 'GDX', 'FETH'
}

# Major indices for market overview
MAJOR_INDICES = ['SPY', 'QQQ', 'IWM']

# Correlation ETFs
CORRELATION_ETFS = ['FNGD', 'FNGU', 'MAGS']
