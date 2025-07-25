@safe_calculation_wrapper
def calculate_breakout_breakdown_analysis(show_debug=False):
    """Calculate breakout/breakdown ratios for major indices"""
    try:
        indices = ['SPY', 'QQQ', 'IWM']
        results = {}
        
        for index in indices:
            try:
                if show_debug:
                    st.write(f"📊 Analyzing breakouts/breakdowns for {index}...")
                
                # Get recent data (3 months for reliable signals)
                ticker = yf.Ticker(index)
                data = ticker.history(period='3mo')
                
                if len(data) < 50:
                    continue
                    
                current_price = data['Close'].iloc[-1]
                
                # Multi-timeframe resistance levels
                resistance_10 = data['High'].rolling(10).max().iloc[-2]   # 10-day high
                resistance_20 = data['High'].rolling(20).max().iloc[-2]   # 20-day high
                resistance_50 = data['High'].rolling(50).max().iloc[-2]   # 50-day high
                
                # Multi-timeframe support levels  
                support_10 = data['Low'].rolling(10).min().iloc[-2]       # 10-day low
                support_20 = data['Low'].rolling(20).min().iloc[-2]       # 20-day low
                support_50 = data['Low'].rolling(50).min().iloc[-2]       # 50-day low
                
                # Breakout signals (price above resistance)
                breakout_10 = 1 if current_price > resistance_10 else 0
                breakout_20 = 1 if current_price > resistance_20 else 0
                breakout_50 = 1 if current_price > resistance_50 else 0
                
                # Breakdown signals (price below support)
                breakdown_10 = 1 if current_price < support_10 else 0
                breakdown_20 = 1 if current_price < support_20 else 0
                breakdown_50 = 1 if current_price < support_50 else 0
                
                # Calculate ratios
                total_breakouts = breakout_10 + breakout_20 + breakout_50
                total_breakdowns = breakdown_10 + breakdown_20 + breakdown_50
                
                breakout_ratio = (total_breakouts / 3) * 100  # Percentage of timeframes showing breakout
                breakdown_ratio = (total_breakdowns / 3) * 100  # Percentage of timeframes showing breakdown
                net_ratio = breakout_ratio - breakdown_ratio  # Net bias
                
                # Determine overall signal strength
                if net_ratio > 66:
                    signal_strength = "Very Bullish"
                elif net_ratio > 33:
                    signal_strength = "Bullish" 
                elif net_ratio > -33:
                    signal_strength = "Neutral"
                elif net_ratio > -66:
                    signal_strength = "Bearish"
                else:
                    signal_strength = "Very Bearish"
                
                results[index] = {
                    'current_price': round(current_price, 2),
                    'breakout_ratio': round(breakout_ratio, 1),
                    'breakdown_ratio': round(breakdown_ratio, 1),
                    'net_ratio': round(net_ratio, 1),
                    'signal_strength': signal_strength,
                    'breakout_levels': {
                        '10d': round(resistance_10, 2),
                        '20d': round(resistance_20, 2), 
                        '50d': round(resistance_50, 2)
                    },
                    'breakdown_levels': {
                        '10d': round(support_10, 2),
                        '20d': round(support_20, 2),
                        '50d': round(support_50, 2)
                    },
                    'active_breakouts': [f"{days}d" for days, signal in 
                                       [('10', breakout_10), ('20', breakout_20), ('50', breakout_50)] if signal],
                    'active_breakdowns': [f"{days}d" for days, signal in 
                                        [('10', breakdown_10), ('20', breakdown_20), ('50', breakdown_50)] if signal]
                }
                
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Error analyzing {index}: {e}")
                continue
        
        # Calculate overall market sentiment
        if results:
            overall_breakout = sum([results[idx]['breakout_ratio'] for idx in results]) / len(results)
            overall_breakdown = sum([results[idx]['breakdown_ratio'] for idx in results]) / len(results)
            overall_net = overall_breakout - overall_breakdown
            
            # Market regime classification
            if overall_net > 50:
                market_regime = "🚀 Strong Breakout Environment"
            elif overall_net > 20:
                market_regime = "📈 Bullish Breakout Bias"
            elif overall_net > -20:
                market_regime = "⚖️ Balanced Market"
            elif overall_net > -50:
                market_regime = "📉 Bearish Breakdown Bias"
            else:
                market_regime = "🔻 Strong Breakdown Environment"
            
            results['OVERALL'] = {
                'breakout_ratio': round(overall_breakout, 1),
                'breakdown_ratio': round(overall_breakdown, 1), 
                'net_ratio': round(overall_net, 1),
                'market_regime': market_regime,
                'sample_size': len(results)
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Breakout/breakdown analysis error: {e}")
        return {}
