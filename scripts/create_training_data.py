#!/usr/bin/env python3
"""
This script creates training data for predictive models.
"""
import arrow
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import talib.abstract as ta
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

from freqtrade.arguments import Arguments, TimeRange
from freqtrade.configuration import Configuration
from freqtrade.data import history
from freqtrade.exchange import exchange
from freqtrade.misc import deep_merge_dicts
from freqtrade.vendor.qtpylib import indicators as qtpylib

import logging

logger = logging.getLogger('feature engineering')

DEFAULT_DL_PATH = 'user_data/data'

if '--mode=client' not in sys.argv[1:]:
    arguments = Arguments(sys.argv[1:], 'feature engineering')
else:
    arguments = Arguments([], 'feature engineering')

arguments.common_options()
arguments.download_data_options()

# Do not read the default config if config is not specified
# in the command line options explicitely
args = arguments.parse_args(no_default_config=True)
args.config = args.config or ['config.json']

# Use bittrex as default exchange
exchange_name = args.exchange or 'bittrex'

pairs: List = []

configuration = Configuration(args)
config: Dict[str, Any] = {}

if args.config:
    # Now expecting a list of config filenames here, not a string
    for path in args.config:
        logger.info(f"Using config: {path}...")
        # Merge config options, overwriting old values
        config = deep_merge_dicts(configuration._load_config_file(path), config)

    config['stake_currency'] = ''
    # Ensure we do not use Exchange credentials
    config['exchange']['dry_run'] = True
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    pairs = config['exchange']['pair_whitelist']

    if config.get('ticker_interval'):
        timeframes = args.timeframes or [config.get('ticker_interval')]
    else:
        timeframes = args.timeframes or ['1m', '5m']

else:
    config = {
        'stake_currency': '',
        'dry_run': True,
        'exchange': {
            'name': exchange_name,
            'key': '',
            'secret': '',
            'pair_whitelist': [],
            'ccxt_async_config': {
                'enableRateLimit': True,
                'rateLimit': 200
            }
        }
    }
    timeframes = args.timeframes or ['1m', '5m']

configuration._load_logging_config(config)

if args.config and args.exchange:
    logger.warning("The --exchange option is ignored, "
                   "using exchange settings from the configuration file.")

# Check if the exchange set by the user is supported
configuration.check_exchange(config)

configuration._load_datadir_config(config)

dl_path = Path(config['datadir'])

pairs_file = Path(args.pairs_file) if args.pairs_file else dl_path.joinpath('pairs.json')

if not pairs or args.pairs_file:
    logger.info(f'Reading pairs file "{pairs_file}".')
    # Download pairs from the pairs file if no config is specified
    # or if pairs file is specified explicitely
    if not pairs_file.exists():
        sys.exit(f'No pairs file found with path "{pairs_file}".')

    with pairs_file.open() as file:
        pairs = list(set(json.load(file)))

    pairs.sort()

timerange = TimeRange()
if args.days:
    time_since = arrow.utcnow().shift(days=-args.days).strftime("%Y%m%d")
    timerange = arguments.parse_timerange(f'{time_since}-')

logger.info(f'About to download pairs: {pairs}, intervals: {timeframes} to {dl_path}')

pairs_not_available = []

data_dict = history.load_data(
    datadir=dl_path,
    ticker_interval=timeframes[0],
    pairs=pairs,
    refresh_pairs=False,
    exchange=exchange.Exchange(config),
    timerange=TimeRange(None, None, 0, 0),
    fill_up_missing=False,
    live=False
)


def populate_indicators(df: pd.DataFrame, metadata: str, thresh: float, window: int) -> pd.DataFrame:
    """
    Adds several different TA indicators to the given DataFrame

    Performance Note: For the best performance be frugal on the number of indicators
    you are using. Let uncomment only the indicator you are using in your strategies
    or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
    :param df: Raw data from the exchange and parsed by parse_ticker_dataframe()
    :param metadata: Additional information, like the currently traded pair
    :param thresh: Threshold for buy/sell signal to be created
    :param window: Window of time into the future to look for threshold to be achieved.
    :return: a Dataframe with all mandatory indicators for the strategies
    """

    dataframe = df.copy()
    dataframe[['open', 'high', 'low', 'close']] = dataframe[['open', 'high', 'low', 'close']] * 1000
    # dataframe[['volume']] = preprocessing.StandardScaler().fit_transform(dataframe[['volume']])


    output = dataframe[['date']].copy()

    # Basic Transformations
    # ------------------------------------
    # output['range'] = dataframe['high'] - dataframe['low']


    # Momentum Indicator
    # ------------------------------------

    # ADX
    output['adx'] = ta.ADX(dataframe)
    output['dmi_diff'] = abs(ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe))

    # Awesome oscillator
    output['ao'] = qtpylib.awesome_oscillator(dataframe)

    # Commodity Channel Index: values Oversold:<-100, Overbought:>100
    # output['cci'] = ta.CCI(dataframe)

    # MACD
    macd = ta.MACD(dataframe)
    # output['macd'] = macd['macd']
    # output['macdsignal'] = macd['macdsignal']
    output['macdhist'] = macd['macdhist']

    # MFI
    # output['mfi'] = ta.MFI(dataframe)

    # Minus Directional Indicator / Movement
    # output['minus_dm'] = ta.MINUS_DM(dataframe)
    # output['minus_di'] = ta.MINUS_DI(dataframe)

    # Plus Directional Indicator / Movement
    # output['plus_dm'] = ta.PLUS_DM(dataframe)
    # output['plus_di'] = ta.PLUS_DI(dataframe)


    # ROC
    output['roc4'] = ta.ROC(dataframe, timeperiod=4)
    output['roc8'] = ta.ROC(dataframe, timeperiod=8)
    output['roc24'] = ta.ROC(dataframe, timeperiod=24)
    output['roc96'] = ta.ROC(dataframe, timeperiod=96)

    # RSI
    # output['rsi'] = ta.RSI(dataframe)

    # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    # output['fisher_rsi'] = fishers_inverse(dataframe['rsi'])

    # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
    # output['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

    # Stoch
    stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=7, slowk_matype=0, slowd_period=3)
    output['slowd'] = stoch['slowd']
    output['slow_diff'] = stoch['slowd'] - stoch['slowk']

    # Stoch fast
    # stoch_fast = ta.STOCHF(dataframe)
    # output['fastd'] = stoch_fast['fastd']
    # output['fastk'] = stoch_fast['fastk']

    # Stoch RSI
    # stoch_rsi = ta.STOCHRSI(dataframe)
    # output['fastd_rsi'] = stoch_rsi['fastd']
    # output['fastk_rsi'] = stoch_rsi['fastk']


    # Volume Indicators
    # ------------------------------------

    # Volume Exponential Moving Averages
    output['vema4'] = dataframe['volume'].ewm(span=4, adjust=False).mean()
    output['vema8'] = dataframe['volume'].ewm(span=8, adjust=False).mean()
    output['vema24'] = dataframe['volume'].ewm(span=24, adjust=False).mean()
    output['vema96'] = dataframe['volume'].ewm(span=96, adjust=False).mean()

    # On Balance Volume
    # output['obv'] = ta.OBV(dataframe)

    # Chaikin A/D Oscilator
    output['adosc'] = ta.ADOSC(dataframe)

    # Overlap Studies
    # ------------------------------------

    # Previous Bollinger bands
    # Because ta.BBANDS implementation is broken with small numbers, it actually
    # returns middle band for all the three bands. Switch to qtpylib.bollinger_bands
    # and use middle band instead.
    # output['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']

    # Bollinger bands
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    output['bb_lowerband'] = bollinger['lower'] - dataframe['close']
    output['bb_middleband'] = bollinger['mid'] - dataframe['close']
    output['bb_upperband'] = bollinger['upper'] - dataframe['close']

    # EMA - Exponential Moving Average
    # output['ema4'] = ta.EMA(dataframe, timeperiod=4)
    # output['ema8'] = ta.EMA(dataframe, timeperiod=8)
    # output['ema24'] = ta.EMA(dataframe, timeperiod=24)
    # output['ema96'] = ta.EMA(dataframe, timeperiod=96)

    # SAR Parabol
    # output['sar'] = ta.SAR(dataframe)

    # SMA - Simple Moving Average
    # output['sma'] = ta.SMA(dataframe, timeperiod=40)

    # TEMA - Triple Exponential Moving Average
    # output['tema'] = ta.TEMA(dataframe, timeperiod=9)

    # Cycle Indicator
    # ------------------------------------
    # Hilbert Transform Indicator - SineWave
    # hilbert = ta.HT_SINE(dataframe)
    # output['htsine'] = hilbert['sine']
    # output['htleadsine'] = hilbert['leadsine']

    # Pattern Recognition - Bullish candlestick patterns
    # ------------------------------------

    # Hammer: values [0, 100]
    # output['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

    # Inverted Hammer: values [0, 100]
    # output['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)

    # Dragonfly Doji: values [0, 100]
    # output['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)

    # Piercing Line: values [0, 100]
    # output['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]

    # Morningstar: values [0, 100]
    # output['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]

    # Three White Soldiers: values [0, 100]
    # output['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]


    # Pattern Recognition - Bearish candlestick patterns
    # ------------------------------------

    # Hanging Man: values [0, 100]
    # output['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)

    # Shooting Star: values [0, 100]
    # output['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)

    # Gravestone Doji: values [0, 100]
    # output['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)

    # Dark Cloud Cover: values [0, 100]
    # output['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)

    # Evening Doji Star: values [0, 100]
    # output['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)

    # Evening Star: values [0, 100]
    # output['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)


    # Pattern Recognition - Bullish/Bearish candlestick patterns
    # ------------------------------------

    # Three Line Strike: values [0, -100, 100]
    # output['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)

    # Spinning Top: values [0, -100, 100]
    # output['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]

    # Engulfing: values [0, -100, 100]
    # output['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]

    # Harami: values [0, -100, 100]
    # output['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]

    # Three Outside Up/Down: values [0, -100, 100]
    # output['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]

    # Three Inside Up/Down: values [0, -100, 100]
    # output['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]


    # Chart type
    # ------------------------------------

    # Heikinashi stategy
    # heikinashi = qtpylib.heikinashi(dataframe)
    # output['ha_open'] = heikinashi['open']
    # output['ha_close'] = heikinashi['close']
    # output['ha_high'] = heikinashi['high']
    # output['ha_low'] = heikinashi['low']

    output = output.set_index('date')
    output = pd.DataFrame(
        data=preprocessing.StandardScaler().fit_transform(output),
        index=output.index,
        columns=output.columns
    )

    output = create_lags(output, 6, 3)
    output['currency_pair'] = metadata.replace('/', '_')
    # if metadata not in pairs[0]:
    #     output[metadata.replace('/', '_')] = 1

    return create_signal(output, df, thresh, window).dropna()

def create_lags(df, lags, step):
    df_list = [df]
    for i in range(1, lags + 1):
        n = i * step
        new_df = df.shift(n)
        new_df.columns = ["{}{}".format(column, "_Lag{:02d}".format(n)) for column in df.columns]
        df_list.append(new_df)
    return pd.concat(df_list, axis=1)

def create_signal(df_x: pd.DataFrame, df_y: pd.DataFrame, thresh: float, window: int) -> pd.DataFrame:
    output = df_y[['date', 'close']].copy().sort_values('date', ascending=False)
    output['signal_val'] = output['close'].ewm(span=window, adjust=True).mean()
    if thresh > 0:
        output['signal_bool'] = np.where(output['signal_val'] / output['close'] - 1 >= thresh, 1, 0)
    else:
        output['signal_bool'] = np.where(output['signal_val'] / output['close'] - 1 <= thresh, 1, 0)
    output = df_x.join(output.sort_values('date', ascending=False).set_index('date'))
    return output.drop(['close', 'signal_val'], axis=1)



train_buy_dict = {
    k : populate_indicators(v, k, .0045, 14) for (k, v) in data_dict.items()
}

train_buy_data = pd.concat(
    objs=train_buy_dict.values(),
    axis=0,
    ignore_index=True,
    sort=False
)

buy_train, buy_test = model_selection.train_test_split(train_buy_data, test_size=0.2)
buy_train, buy_val = model_selection.train_test_split(buy_train, test_size=0.2)

for df_tup in [('buy_train', buy_train), ('buy_test', buy_test), ('buy_val', buy_val)]:
    df_tup[1].to_csv(
        path_or_buf='user_data/data/training/{}.csv'.format(df_tup[0]),
        header=True,
        index=False,
        compression='infer'
    )

# train_sell_dict = {
#     k : create_sell_signal(create_lags(populate_indicators(v), 6, 3), v, -.004, 12)
#         .dropna() for (k, v) in data_dict.items()
# }
