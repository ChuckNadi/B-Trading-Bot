#BTB9

# Average Buy Fee per Trade: 0.09096296296296297 USDT or  For buying transactions: approximately 0.784%
# Average Sell Fee per Trade: 0.2321666285714286 USDT or  For selling transactions: approximately 0.219%

    
import pandas as pd
import numpy as np
from binance.client import Client
import os
import time
from binance.exceptions import BinanceAPIException
import logging
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename="C:\\Users\\charl\\Desktop\\wbb.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
import json
import requests
import matplotlib.pyplot as plt

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, tld='us')

class TradingBot:


    #for manual sell prices for specific (applies mainly for first sell-condition)
  specific_percentage_increases = {
      'BTCUSDT': 1.5,  # 2% increase for BTCUSDT before sell
      'ETHUSDT': 1.8,  # 1.5% increase for ETHUSDT before sell
      'SOLUSDT': 3.5,
}
  default_percentage_increase = 3  # Default increase for other symbols 

  # for trailing stop loss percentages for specific symbols (applies mainly for trailing stop loss sell-condition)
  specific_stop_loss_percentages = {
      'BTCUSDT': 4,  # 3% stop loss for BTCUSDT
      'ETHUSDT': 5,  # 3.6% stop loss for ETHUSDT
      'SOLUSDT': 6,
        # Add other symbols as needed
  }
  default_stop_loss_percentage = 7 # Default stop loss percentage for other symbols

  def __init__(self, symbols, quantities, compound_factor=0.5, max_quantity=200):
    self.symbols = symbols
    self.quantities = quantities
    self.prices = {}
    self.bought_orders = {} 
    self.highest_prices = {} 
    self.state_file = "C:\\Users\\charl\\Desktop\\trading_bot_state.json"  # Choose an appropriate file path
    self.load_state()    
    self.highest_since_buy = {} 


    # Performance Metrics
    self.total_trades = 0
    self.winning_trades = 0
    self.losing_trades = 0
    self.total_fees = 0
    self.total_profit_loss = 0
    self.decision_log = []
    self.compound_factor = compound_factor
    self.max_quantity = max_quantity

    # Load historical klines
    self.history = {}
    for symbol in self.symbols:
      self.history[symbol] = self.load_historical_data(symbol)

  def initialize_manual_orders(self):
        """Initializes the bot with manual orders."""
        manual_orders = {
            'ETHUSDT': [{'price': 3518.0, 'quantity': 0.1}, {'price': 3727.72, 'quantity': 0.1}, {'price': 3727.72, 'quantity': 0.1}],
            'XRPUSDT': [{'price': 0.6063000000000001, 'quantity': 100.0}, {'price': 0.6455, 'quantity': 100.0}, {'price': 0.645637, 'quantity': 100.0}],
            'BTCUSDT': [{'price': 66660.992, 'quantity': 0.01}, {'price': 67654.56661, 'quantity': 0.01}, {'price': 67665.24, 'quantity': 0.01}],
            'SOLUSDT': [{'price': 174.9975142857143, 'quantity': 7.0}, {'price': 192.17, 'quantity': 7.0}, {'price': 192.33002857142856, 'quantity': 7.0}]
        }
        for symbol, orders in manual_orders.items():
            if symbol not in self.bought_orders or not self.bought_orders[symbol]:
                self.bought_orders[symbol] = orders
            else:
                self.bought_orders[symbol].extend(orders)


  def get_overall_wallet_balance(self):
      # Fetch all asset balances
    balances = client.get_account()['balances']

    # Calculate the total balance in USDT
    total_balance_in_usdt = 0.0
    for balance in balances:
        asset = balance['asset']
        free_balance = float(balance['free'])
        locked_balance = float(balance['locked'])
        total_balance = free_balance + locked_balance

        # If the asset is not USDT, convert its balance to USDT
        if asset != 'USDT' and total_balance > 0:
            try:
                conversion_rate = float(client.get_symbol_ticker(symbol=f"{asset}USDT")["price"])
                total_balance_in_usdt += total_balance * conversion_rate
            except BinanceAPIException:
                # Handle the case where there's no direct USDT pair for the asset
                pass
        else:
            total_balance_in_usdt += total_balance

    return total_balance_in_usdt
    
# # When a buy order is executed
#   def execute_buy_order(self, symbol, price, desired_quantity, fee_percentage, order_id):
#       # Calculate the total cost without fees
#       total_cost_without_fee = price * desired_quantity

#       # Adjust for the fees
#       fee = total_cost_without_fee * fee_percentage / 100
#       total_cost_with_fee = total_cost_without_fee + fee

#       # Calculate the adjusted quantity to ensure desired quantity is received after fees
#       adjusted_quantity = total_cost_with_fee / price

#       # Execute the buy order
#       order_response = client.create_order(
#           symbol=symbol,
#           side='BUY',
#           type='MARKET',
#           quantity=adjusted_quantity
#       )

#       # Record the order details
#       order_details = {
#           'id': order_id,
#           'price': price,
#           'quantity': desired_quantity,  # Original desired quantity
#           'usdt_amount': total_cost_with_fee
#       }
#       self.bought_orders.setdefault(symbol, []).append(order_details)
#       return order_response
  








  def execute_buy_order(self, symbol, predictions_data):
        print("Executing Buy Order for", symbol)

        # Extract last_buy_price from predictions_data
        last_buy_price = predictions_data.get('last_buy_price', None)
        # total_cost_without_fee = price * desired_quantity
        # fee = total_cost_without_fee * fee_percentage / 100



        # Check and handle different data types
        if isinstance(last_buy_price, (int, float)):
            predicted_price = last_buy_price
        elif isinstance(last_buy_price, (list, pd.Series, np.ndarray)):
            predicted_price = last_buy_price[-1]  # Get the last item if it's a list or array-like
        else:
            print("Error: last_buy_price format not recognized", last_buy_price)
            return None  # Exit the function if the format is not recognized

        # Assuming 'available_balance' and 'client' are accessible within this context
        available_balance = self.get_overall_wallet_balance() 
        if available_balance <= 0:
            print("Insufficient balance to place order.")
            return None

        # Calculate the quantity to buy based on the predicted price
        quantity_to_buy = available_balance / predicted_price
        try:
            # Execute the buy order using the Binance client or similar
            order_response = client.create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity_to_buy
            )

            self.bought_orders.setdefault(symbol, []).append(order_response)

            # Process order response and update internal state as necessary
            print(f"Buy order executed successfully: {order_response}")
            return order_response
        except Exception as e:
            print(f"Failed to execute buy order: {e}")
            return None

  # def execute_buy_order(self, symbol, price, quantity, order_id):
    # usdt_amount = price * quantity
    # order_details = {'id': order_id, 'price': price, 'quantity': quantity, 'usdt_amount': usdt_amount}
    # self.bought_orders.setdefault(symbol, []).append(order_details)

  # When a sell order is executed
  def execute_sell_order(self, symbol, sell_price, sell_quantity, order_id):
      # Find the matching buy order by ID
      matching_order = next((order for order in self.bought_orders[symbol] if order['id'] == order_id), None)
      
      if matching_order:
          # Calculate the total value and the fee
          total_value = matching_order['quantity'] * current_price
          fee = total_value * fee_percentage / 100

          # Adjust the sell quantity to cover the fee
          net_value = total_value - fee
          adjusted_quantity = net_value / current_price

          # Place the sell order with the adjusted quantity
          order_response = client.create_order(
              symbol=symbol, 
              side='SELL', 
              type='MARKET',                                 
              quantity=adjusted_quantity,
          )
      # if matching_order:
          # quantity_to_sell = matching_order['quantity']
          bought_price = matching_order['price']
          actual_profit = (sell_price - bought_price) * adjusted_quantity
          self.log_trade(..., actual_profit, ...)
          self.bought_orders[symbol].remove(matching_order)
          
                # Clean up if no more orders for the symbol
          if not self.bought_orders[symbol]:
              del self.bought_orders[symbol]
      else:
          print(f"No matching buy order found for sell order ID: {order_id}")





          
  def log_trade(self, wallet_initial_state, symbol, action, current_price, quantity, fee, predicted_profit, actual_profit, order):
      total_balance = self.get_overall_wallet_balance()

      # Compute the actual profit or loss
      if action == "sell":
          # Ensure 'order' is valid and contains the 'price' key
          if order and 'price' in order:
              bought_price = float(order['price'])
              actual_profit = (current_price - bought_price) * quantity - fee
          else:
              actual_profit = 0  # Set to 0 or handle appropriately if 'order' is invalid
      elif action == "buy":
          # For buy orders, 'actual_profit' can be set to 0 or calculated differently
          actual_profit = 0

      # Log the trade details
      logging.info(
          f"Trade executed: Wallet Initial State: {wallet_initial_state}, Symbol: {symbol}, Action: {action}, Current Price: {current_price}, Quantity: {quantity}, Fee: {fee}, Predicted Profit: {predicted_profit}, Actual Profit: {actual_profit}, Overall Wallet Balance: {total_balance:.2f} USDT"
      )

      # Update trade statistics
      self.total_trades += 1
      self.total_fees += fee
      self.total_profit_loss += actual_profit
        

        
          

  # def log_trade(self, wallet_initial_state, symbol, action, current_price, quantity, fee, predicted_profit, actual_profit, bought_price=None):
    # total_balance = self.get_overall_wallet_balance()

    # # Compute the actual profit or loss
    # if action == "sell": # and bought_price is not None:
        # #logging.info(f"Starting trade cycle with wallet_initial_state: {wallet_initial_state}. Trade for {symbol}. Action: {action}, Current Price: {current_price}, Quantity: {quantity}, Fee: {fee}, Predicted Profit: {predicted_profit}, Actual Profit: {actual_profit}, Overall Wallet Balance: {total_balance:.2f} USDT") 
        # #actual_profit = (current_price - bought_price) * quantity - fee
        # actual_profit = (float(client.get_symbol_ticker(symbol=symbol)["price"]) - float(order['price'])) * quantity - fee
        # logging.info(f"Trade executed: Wallet Initial State: {wallet_initial_state}, Symbol: {symbol}, Action: {action}, Current Price: {current_price}, Quantity: {quantity}, Fee: {fee}, Predicted Profit: {predicted_profit}, Actual Profit: {actual_profit}, Overall Wallet Balance: {total_balance:.2f} USDT")        
        # #actual_profit = (current_price - bought_price) * quantity - fee


     # #########################   
    # if actual_profit > 0:
        # self.adjust_trade_quantity(symbol, actual_profit, current_price)        

        # self.winning_trades += 1
    # elif action == "buy":
        # logging.info(
            # f"Starting trade cycle with wallet_initial_state: {wallet_initial_state}. Trade for {symbol}. Action: {action}, Current Price: {current_price}, Quantity: {quantity}, Fee: {fee}, Predicted Profit: {predicted_profit}, Actual Profit: {actual_profit}, Overall Wallet Balance: {total_balance:.2f} USDT")
    # else:
        # actual_profit = 0  # Or any other default value for non-sell actions

    # self.total_trades += 1
    # self.total_fees += fee
    # self.total_profit_loss += actual_profit

  def analyze_trade(self, symbol, current_price, predicted_profit, actual_profit):
    # Trade Analysis
    if actual_profit < 0:
        logging.warning(f"Loss detected for {symbol}. Predicted Profit: {predicted_profit}, Actual Profit: {actual_profit}, Current Price: {current_price}")
    # Further analysis can be added here

  def log_decision_making(self, symbol, indicators):
    # Decision-Making Analysis
    self.decision_log.append({
        'symbol': symbol,
        'indicators': indicators  # e.g. {'ema': value, 'rsi': value, ...}
    })
    logging.info(f"Decision Making for {symbol}. Indicators: {indicators}")

  def visualize_trades(self):
    # Assuming you are logging trade details in a list (e.g., self.trades_log)
    # where each entry is a dictionary with keys: ['symbol', 'action', 'price', 'quantity', 'profit']

    # Extract trade details
    prices = [trade['price'] for trade in self.trades_log]
    profits = [trade['profit'] for trade in self.trades_log]
    actions = [trade['action'] for trade in self.trades_log]

    # Plotting prices
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Trade Prices', color='blue')
    plt.xlabel('Trade Number')
    plt.ylabel('Price')
    plt.title('Trade Prices Over Time')
    plt.legend()
    plt.show()

    # Plotting profits
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(profits)), profits, color=['green' if action == 'buy' else 'red' for action in actions])
    plt.xlabel('Trade Number')
    plt.ylabel('Profit/Loss')
    plt.title('Profit/Loss Per Trade')
    plt.show()

  def load_historical_data(self, symbol):
    #For the bot to retry when internet connection is bad or api calls fails.
    RETRY_COUNT = 3
    RETRY_DELAY = 10  # 5 seconds delay

    for i in range(RETRY_COUNT):
        try:
            bars_15m = client.get_klines(symbol=symbol, interval='15m', limit=300)
            # If the call is successful, break out of the loop
            break
        #except requests.exceptions.ReadTimeout:   
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # If we've reached the max number of retries, re-raise the exception
            if i == RETRY_COUNT - 1:
                raise
            # Otherwise, wait and then try again
            else:
            #time.sleep(RETRY_DELAY)
                time.sleep(RETRY_DELAY * (2 ** i))  # Exponential backoff


    # Fetch last 300 4h candles 
    col_names = ['time', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']    

    bars_15m = client.get_klines(symbol=symbol, interval='15m', limit=300)    
    bars_30m = client.get_klines(symbol=symbol, interval='30m', limit=300)   
    bars_1h = client.get_klines(symbol=symbol, interval='1h', limit=300)  
    bars_4h = client.get_klines(symbol=symbol, interval='4h', limit=300) 

    df_15m = pd.DataFrame(bars_15m, columns=col_names)
    df_30m = pd.DataFrame(bars_30m, columns=col_names)
    df_1h = pd.DataFrame(bars_1h, columns=col_names)
    df_4h = pd.DataFrame(bars_4h, columns=col_names)

    df_15m['close'] = df_15m['close'].astype(float) 
    df_30m['close'] = df_30m['close'].astype(float)
    df_1h['close'] = df_1h['close'].astype(float)  
    df_4h['close'] = df_4h['close'].astype(float)

    df_15m['open'] = df_15m['open'].astype(float) 
    df_30m['open'] = df_30m['open'].astype(float)
    df_1h['open'] = df_1h['open'].astype(float)  
    df_4h['open'] = df_4h['open'].astype(float)   

    df_15m['high'] = df_15m['high'].astype(float) 
    df_30m['high'] = df_30m['high'].astype(float)
    df_1h['high'] = df_1h['high'].astype(float)  
    df_4h['high'] = df_4h['high'].astype(float) 

    df_15m['low'] = df_15m['low'].astype(float) 
    df_30m['low'] = df_30m['low'].astype(float)
    df_1h['low'] = df_1h['low'].astype(float)  
    df_4h['low'] = df_4h['low'].astype(float) 

    df_15m['volume'] = pd.to_numeric(df_15m['volume'], errors='coerce')
    df_30m['volume'] = pd.to_numeric(df_30m['volume'], errors='coerce')
    df_1h['volume'] = pd.to_numeric(df_1h['volume'], errors='coerce')
    df_4h['volume'] = pd.to_numeric(df_4h['volume'], errors='coerce')

    df_15m['time'] = pd.to_datetime(df_15m['time'], unit='ms')
    df_30m['time'] = pd.to_datetime(df_30m['time'], unit='ms')
    df_1h['time'] = pd.to_datetime(df_1h['time'], unit='ms')
    df_4h['time'] = pd.to_datetime(df_4h['time'], unit='ms')

    df_15m = df_15m.set_index('time')
    df_30m = df_30m.set_index('time')
    df_1h = df_1h.set_index('time')
    df_4h = df_4h.set_index('time')

    return df_15m, df_30m, df_1h, df_4h

  def EMA(self, data, period=20):
    return data.rolling(window=period).mean()  

  def RSI(self, data, period=14):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.rolling(window=period).mean() 
    avg_loss = abs(down.rolling(window=period).mean())

    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1 + rs))
    return rsi

  def MACD(self, data):
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

  def stoch_rsi(self, rsi, period=14):
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stochrsi * 100

  ################################################
  def adjust_trade_quantity(self, symbol, profit, current_price):
    additional_quantity = (profit * self.compound_factor) / current_price
    if additional_quantity + self.quantities[symbol] <= self.max_quantity:
        self.quantities[symbol] += additional_quantity
    else:
        self.quantities[symbol] = self.max_quantity

  def analyze(self, symbol, df):  
    ema20 = self.EMA(df['close'], 20)
    ema50 = self.EMA(df['close'], 50)
    rsi = self.RSI(df['close'])
    macd, signal = self.MACD(df['close'])
    stoch_rsi = self.stoch_rsi(rsi)

    analysis = {}

    # trend
    if ema20.iloc[-1] > ema50.iloc[-1]:      
        analysis['trend'] = 'up'
    else:
        analysis['trend'] = 'down'
    ####################################################################################
    # print(f"Analyzing {symbol}")
    # print(f"EMA20: {ema20.iloc[-1]}, EMA50: {ema50.iloc[-1]}")
    # print(f"RSI: {rsi.iloc[-1]}, Stoch RSI: {stoch_rsi.iloc[-1]}") 
    # print(f"MACD: {macd.iloc[-1]}, Signal: {signal.iloc[-1]}")
    # print(analysis)    
    ####################################################################################


    # Bullish Divergence
    price_low = df['low'].iloc[-2] > df['low'].iloc[-1]   # price making a lower low
    rsi_high = rsi.iloc[-2] < rsi.iloc[-1]                # RSI making a higher low
    bullish_divergence = price_low and rsi_high

    # Previous Support (using last 50 candles for support)
    support_level = df['close'].tail(50).min()
    at_support = df['close'].iloc[-1] < support_level * 1.03 and df['close'].iloc[-1] > support_level * 0.97

    # Candlestick Confirmation (Bullish Engulfing)
    open_prev = df['open'].iloc[-2]
    close_prev = df['close'].iloc[-2]
    open_curr = df['open'].iloc[-1]
    close_curr = df['close'].iloc[-1]
    bullish_engulfing = open_curr < close_prev and close_curr > open_prev

    # Volume Increase (using last 50 candles for volume average)
    avg_volume = df['volume'].tail(50).mean()
    volume_increase = df['volume'].iloc[-1] > 1.5 * avg_volume

    # Entries
    if (analysis['trend'] == 'up' and bullish_divergence and at_support and bullish_engulfing and volume_increase):
        analysis['entry'] = 'buy'
    elif rsi.iloc[-1] > 70 and stoch_rsi.iloc[-1] > 80 and macd.iloc[-1] < signal.iloc[-1]:      
        analysis['entry'] = 'sell'

    return {
        'analysis': analysis, 'macd': macd.iloc[-1],'signal': signal.iloc[-1]}


  def load_state(self):
      try:
          with open(self.state_file, 'r') as f:
              state = json.load(f)
              self.bought_orders = state.get('bought_orders', {})
              self.highest_prices = state.get('highest_prices', {})
      except FileNotFoundError:
          self.bought_orders = {}
          self.highest_prices = {}


        # Ensure there are no empty lists in self.bought_orders
      for symbol in list(self.bought_orders.keys()):
          if not self.bought_orders[symbol]:
              del self.bought_orders[symbol]       


  def save_state(self):
    state = {
        'bought_orders': self.bought_orders,
        'highest_prices': self.highest_prices
    }
    with open(self.state_file, 'w') as f:
        json.dump(state, f)

        # Log or print a confirmation message
    #logging.info("State saved successfully.")
    
    
  def run(self):

    ##

    print(f"Starting trading for symbols: {self.symbols}") 
    actions = []


    for symbol in self.symbols:
            # Fetch the latest price for the symbol
        current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
        #print(f"Raw price data for {symbol}: {client.get_symbol_ticker(symbol=symbol)}")

        # Update the highest price for this symbol if it's higher than the recorded highest price
        self.highest_prices[symbol] = max(self.highest_prices.get(symbol, 0), current_price)
        # Save the state after updating the highest price
        self.save_state()

        action = None  # Resetting the action variable for each symbol
        df_15m, df_30m, df_1h, df_4h = self.history[symbol]        
        analysis_15m = self.analyze(symbol, df_15m)
        analysis_30m = self.analyze(symbol, df_30m)
        analysis_1h = self.analyze(symbol, df_1h)
        analysis_4h = self.analyze(symbol, df_4h)

        # Count how many timeframes have an upward trend
        upward_trends = sum([1 for analysis in [analysis_15m['analysis'], analysis_30m['analysis'], analysis_1h['analysis'], analysis_4h['analysis']] if analysis['trend'] == 'up'])
        downward_trends = sum([1 for analysis in [analysis_15m['analysis'], analysis_30m['analysis'], analysis_1h['analysis'], analysis_4h['analysis']] if analysis['trend'] == 'down'])

        # Decide on the action based on the number of upward trends (order of operations)
        action = None


        if upward_trends >= 2 and analysis_15m['macd'] > analysis_15m['signal']: # at least 2 timeframes are trending upwards
            print(f"Buy condition for {symbol} met.")
            action = 'buy'           

        elif downward_trends >= 2 or (symbol in self.bought_orders and self.bought_orders[symbol] and float(client.get_symbol_ticker(symbol=symbol)["price"]) >= 1.03 * self.bought_orders[symbol][0]['price']):
            print(f"Sell condition {symbol}) met.")
            action = 'sell'
 
        # if analysis['entry'] == 'sell' and symbol in self.bought_orders and self.bought_orders[symbol] and float(client.get_symbol_ticker(symbol=symbol)["price"]) >= 1.03 * self.bought_orders[symbol][0]['price']:
            # print(f"Sell condition for {symbol} met.")
            # action = 'sell'



        if action:
            actions.append((symbol, action))



    for symbol, action in actions:
        # Fetch the latest price for the symbol
        current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])


        # New logic to fetch bought price for the symbol
        if symbol in self.bought_orders and self.bought_orders[symbol]:
            # Assuming each symbol has only one order for simplicity.
            bought_price = self.bought_orders[symbol][0]['price']

        else:
            print(f"No orders found for {symbol}.")
            bought_price = None 

        # # Calculate percentage_change only if bought_price is not zero
        if bought_price is not None:
            percentage_change = ((current_price - bought_price) / bought_price) * 100
            specific_increase = self.specific_percentage_increases.get(symbol, self.default_percentage_increase)            
            
            if percentage_change >= specific_increase:
                print(f"--------{symbol} {percentage_change:.5f}% increase from bought price.--------")
            else:
                print(f"--------{symbol} {-percentage_change:.5f}% decrease from bought price.--------")

        # Existing line
        print(f"{symbol} Current Price: {current_price}, Bought Price: {bought_price}, Highest Price: {self.highest_prices.get(symbol, 0)}")


        if action == 'buy':
            predicted_profit = 0  # No profit prediction when buying

            try:
#######################################################################################################            
                # # Check if there are already 3 open buy orders for the symbol if not skip buy order execution
                if symbol in self.bought_orders and len(self.bought_orders[symbol]) >= 3:
                    print(f"Skip buying {symbol}: Maximum buy orders reached for {symbol}.")
                    continue

#######################################################################################################
                #Executing placing buy order
                order = client.create_order(
                    symbol=symbol, 
                    side=action, 
                    type='MARKET', 
                    quantity=self.quantities[symbol])

                print(f"Placed {action} order for {self.quantities[symbol]} {symbol} - {order['orderId']}")
                print(order)

                if order['status'] == 'FILLED':                               
                    fee = float(order['fills'][0]['commission'])
                    fee_asset = order['fills'][0]['commissionAsset']



#######################################################################################################
                    cummulative_quote_qty = float(order['cummulativeQuoteQty'])
                    quantity_executed = float(order['executedQty'])
                    bought_price = cummulative_quote_qty / quantity_executed  # Includes fees
                    #self.bought_orders[symbol].append({'price': bought_price, 'quantity': quantity_executed})
                    #self.bought_orders.setdefault(symbol, []).append(order_details)
                    self.bought_orders.setdefault(symbol, []).append({'price': bought_price, 'quantity': quantity_executed})

########################################################################3##    ########################
                else:
                    fee = 0.0                

                # Update the bought_orders dictionary
####################################################################################################### 
                # THIS UPDATES THE ORDERS SO YOU CAN SEE NEXT TIME IT CHECKS IT ENSURES THERE IS ONLY 3 ORDERS PRESENT.
                # order_details = {'price': current_price, 'quantity': self.quantities[symbol]}
                # self.bought_orders.setdefault(symbol, []).append(order_details)                

#######################################################################################################                
                order_details = {'price': current_price, 'quantity': self.quantities[symbol]}
                actual_profit = 0
                wallet_initial_state = self.get_overall_wallet_balance()
                self.log_trade(wallet_initial_state, symbol, action, current_price, self.quantities[symbol], fee, predicted_profit, actual_profit, order )

                if symbol in self.bought_orders:
                    self.bought_orders[symbol].append(order_details)
                else:
                    self.bought_orders[symbol] = [order_details]        

                # Save the updated state after placing the order
                self.save_state()            

            except BinanceAPIException as e:
                print(f"Error placing {action} order for {symbol}: {str(e)}")
                self.highest_prices[symbol] = max(self.highest_prices.get(symbol, 0), current_price)



         # SELL Condition: Sell if current price is AT LEAST 3% HIGHER than bought price
        if action == 'sell' and symbol in self.bought_orders:
            # Predict a 1% increase from the last buy price as profit
            last_buy_price = self.bought_orders[symbol][-1]['price']
            predicted_profit = 0.03 * last_buy_price * self.quantities[symbol]  # This assumes you're selling the entire quantity
            specific_increase = self.specific_percentage_increases.get(symbol, self.default_percentage_increase)
            necessary_sell_price = last_buy_price * (1 + specific_increase / 100)

            self.bought_orders[symbol].sort(key=lambda x: x['price'])        

            #for order in self.bought_orders[symbol
            for order_index, order in enumerate(self.bought_orders[symbol]):        

                bought_price = self.bought_orders[symbol][0]['price']
                highest_price = self.highest_prices.get(symbol, 0)

                # Print out the current, bought, and highest prices
                print(f"Order {order_index + 1} for {symbol}: Bought Price = {bought_price}, Selling Aim >= {bought_price * 1.03}, Highest Price: {highest_price}")        


                # Check sell conditions based on the individual bought price (FIRST CONDITION)
                #if (current_price >= 1.03 * bought_price) and (current_price <= 0.995 * highest_price):
                if current_price >= necessary_sell_price:

                    print(f"PASSED 1:{symbol} 3% REACHED. Placing sell order.")
                    stop_loss_price = bought_price * 0.998  # 2% below bought price  

                    # Fetch the balance for the asset
                    asset_balance = client.get_asset_balance(asset=symbol[:-4])  # Assuming USDT pairs. e.g., for 'ETHUSDT', it fetches 'ETH'
                    asset_quantity = float(asset_balance['free'])

                    # Calculate the total worth of the asset in USDT
                    asset_worth_in_usdt = asset_quantity * current_price

                    # # Calculate the amount needed to sell in USDT
                    amount_needed_to_sell = float(order['quantity']) * current_price
                    # print(f"%%%% For {symbol} {amount_needed_to_sell:.2f}USDT needed to fulfill sell order. {asset_worth_in_usdt:.2f} USDT worth of {symbol[:-4]} is currently available in the account. %%%%") 
                    
                    #
                    potential_profit_if_sold = asset_worth_in_usdt - amount_needed_to_sell
                    #print(f"%%%%%%%%%%%%%%% For {symbol}, potential profit/Loss if sold: {potential_profit_if_sold:.2f} USDT. {asset_worth_in_usdt:.2f} USDT worth of {symbol[:-4]} is currently available in the account. %%%%%%%%%%%%%%%")
                    print(f"################ For {symbol}, potential profit/Loss if sold: {potential_profit_if_sold:.2f} USDT. "
                          f"{asset_worth_in_usdt:.2f} USDT worth of {symbol[:-4]} is currently available in the account and {necessary_sell_price:.2f} USDT and {specific_increase}% from bought price is needed to sell."
                          "################")                    

                    
                    # Placing the sell order                       

                    try:
                        # Extract quantity from the bought order
                        quantity_to_sell = order['quantity']


                        # Place the sell order
                        order_response = client.create_order(
                            symbol=symbol, 
                            side=action, 
                            type='MARKET',                                 
                            quantity=quantity_to_sell,
                            )
                        print(f"Debug:{symbol} Sell order placed CONDITION 1 passed. Order details: {order_response}")               


          
                        if order_response['status'] == 'FILLED':                               
                            fee = float(order_response['fills'][0]['commission'])
                            fee_asset = order_response['fills'][0]['commissionAsset']

##########################################################################################
                            cummulative_quote_qty = float(order_response['cummulativeQuoteQty'])
                            quantity_executed = float(order_response['executedQty'])
                            bought_price = cummulative_quote_qty / quantity_executed  # Includes fees
##########################################################################################



                        else:
                            fee = 0.0

                        actual_profit = (current_price - bought_price) * order['quantity']  # Simplified
                        wallet_initial_state = self.get_overall_wallet_balance()
                        self.log_trade(wallet_initial_state, symbol, action, current_price, self.quantities[symbol], fee, predicted_profit, actual_profit, order)
                        self.analyze_trade(symbol, current_price, predicted_profit, actual_profit)                

                        if 'orderId' in order:        
                            print(f"Placed {action} order for {self.quantities[symbol]} {symbol} - {order_response['orderId']}")
                            print(order)
                            
                            ########### After a successful sell#############
                            profit = (current_price - bought_price) * order['quantity']
                            #profit = actual_profit * 0.5 # compound 50% of profit
                            self.adjust_trade_quantity(symbol, profit, current_price)
                            
                        else:
                            print(f"Failed to place {action} order for {symbol}. Response: {order}")
                        # Once the order is sold, remove it from the list
                        self.bought_orders[symbol].remove(order)
                        self.save_state()

                        # If there are no more orders for this symbol, remove the symbol from the dictionary
                        if not self.bought_orders[symbol]:
                            del self.bought_orders[symbol]
                        print(f"Successfully sold {symbol}. It's no longer in the bought_orders list.")

                        # Immediate check after selling
                        if symbol in self.bought_orders:
                            print(f"Error: {symbol} was just sold but still exists in bought_orders list.")
                        else:
                            print(f"Successfully sold {symbol}. It's no longer in the bought_orders list.")    


                        # Only sell one order at a time for each run
                        break

                    except BinanceAPIException as e:
                        print(f"Error placing {action} order for {symbol}: {str(e)}") 
                else:
                    #print(f"Debug:{symbol} CONDITION 1 failed, Current price {current_price} is not higher than 3% of bought price and has not fallen more than 2% from Highest price {highest_price}.")    
                    print(f"Debug: {symbol}, CONDITION 1 failed: Current price {current_price:.2f} USDT is not higher than {specific_increase}% of the bought price which will be {bought_price * (1 + specific_increase / 100):.2f} USDT and has not fallen more than 7% from the bought price {bought_price:.2f} USDT.")



                #Adjusted trailing stop loss condition: Sell if current price is 7% below the bought price
                if symbol in self.bought_orders and self.bought_orders[symbol]:
                    bought_price = self.bought_orders[symbol][0]['price']  # Assuming the first order's price is the bought price
                    #trailing_stop_loss_price = bought_price * 0.93  # 7% below bought price
                    
                            # Get the specific stop loss percentage or use the default
                    stop_loss_percentage = self.specific_stop_loss_percentages.get(symbol, self.default_stop_loss_percentage)
                    trailing_stop_loss_price = bought_price * (1 - stop_loss_percentage / 100)            

                if current_price <= trailing_stop_loss_price:
                    #print(f"Debug: Adjusted trailing stop loss condition met for {symbol}. Placing sell order.")
                    print(f"Debug: Adjusted trailing stop loss condition met for {symbol}. {symbol} is below {stop_loss_percentage}% of the bought price. Placing sell order.")                    
                    stop_loss_price = trailing_stop_loss_price  # 7% below the bought price




                    # Fetch the balance for the asset
                    asset_balance = client.get_asset_balance(asset=symbol[:-4])  # Assuming USDT pairs. e.g., for 'ETHUSDT', it fetches 'ETH'
                    asset_quantity = float(asset_balance['free'])
                    asset_worth_in_usdt = asset_quantity * current_price
                    amount_needed_to_sell = float(order['quantity']) * current_price
                    
               

                    #print(f"%%%% For {symbol} {amount_needed_to_sell:.2f}USDT needed to fulfill sell order. {asset_worth_in_usdt:.2f} USDT worth of {symbol[:-4]} is currently available in the account. %%%%%") 
                    
                    potential_profit_if_sold = asset_worth_in_usdt - amount_needed_to_sell
                    print(f"%%%%%%%%%%%%%%% For {symbol}, potential profit/Loss if sold: {potential_profit_if_sold:.2f} USDT. {asset_worth_in_usdt:.2f} USDT worth of {symbol[:-4]} is currently available in the account. %%%%%%%%%%%%%%%")

                    #Placing the sell order         

                    try:
                        # Extract quantity from the bought order
                        quantity_to_sell = order['quantity']
                        

                        # Place the sell order
                        order_response = client.create_order(
                            symbol=symbol, 
                            side=action, 
                            type='MARKET',                                 
                            quantity=quantity_to_sell,
                        )
                        print(f"Debug:{symbol} Sell order placed. Order details: {order_response}")               
                        # print(f"Placed {action} order for {self.quantities[symbol]} {symbol} - {order_response['orderId']}")
                        # print(order)
                        
                        if order_response['status'] == 'FILLED':                               
                            fee = float(order_response['fills'][0]['commission'])
                            fee_asset = order_response['fills'][0]['commissionAsset']
                            
                            
                            cummulative_quote_qty = float(order_response['cummulativeQuoteQty'])
                            quantity = float(order_response['executedQty'])
                            bought_price = cummulative_quote_qty / quantity  # Includes fee    

                        else:
                            fee = 0.0

                        actual_profit = (current_price - bought_price) * order['quantity']  # Simplified
                        wallet_initial_state = self.get_overall_wallet_balance()
                        self.log_trade(wallet_initial_state, symbol, action, current_price, self.quantities[symbol], fee, predicted_profit, actual_profit, order)
                        self.analyze_trade(symbol, current_price, predicted_profit, actual_profit)                

                        if 'orderId' in order:        
                            print(f"Placed {action} order for {self.quantities[symbol]} {symbol} - {order_response['orderId']}")
                            print(order)
                            ########### After a successful sell#############
                            profit = (current_price - bought_price) * order['quantity']
                            #profit = actual_profit * 0.5 # compound 50% of profit
                            self.adjust_trade_quantity(symbol, profit, current_price)
                        else:
                            print(f"Failed to place {action} order for {symbol}. Response: {order}")
                        # Once the order is sold, remove it from the list
                        self.bought_orders[symbol].remove(order)
                        self.save_state()

                        # If there are no more orders for this symbol, remove the symbol from the dictionary
                        if not self.bought_orders[symbol]:
                            del self.bought_orders[symbol]
                        print(f"Successfully sold {symbol}. It's no longer in the bought_orders list.")

                        # Immediate check after selling
                        if symbol in self.bought_orders:
                            print(f"Error: {symbol} was just sold but still exists in bought_orders list.")
                        else:
                            print(f"Successfully sold {symbol}. It's no longer in the bought_orders list.")    

                        # Only sell one order at a time for each run
                        break

                    except BinanceAPIException as e:
                        print(f"Error placing {action} order for {symbol}: {str(e)}") 
                else:
                    #print(f"Debug:{symbol} STOP LOSS logic failed Current price {current_price} is not fallen 5% of Bought price {bought_price}.")
                    print(f"Debug:{symbol} STOP LOSS logic failed Current price {current_price} is not fallen {stop_loss_percentage}% of Bought price {bought_price}.")
                    #print(f"Debug: {symbol} STOP LOSS logic failed. Current price {current_price:.2f} USDT has not fallen by {stop_loss_percentage}% from the Bought price {bought_price:.2f} USDT.")

        else:
            print(f"Debug: Asset {symbol} wasn't bought before. Skipping sell.")
            # Update the highest price for this symbol
            self.highest_prices[symbol] = max(self.highest_prices.get(symbol, 0), current_price)                     
            predicted_profit = 0

        # New logic to fetch bought price for the symbol
        if symbol in self.bought_orders and self.bought_orders[symbol]:
            # Assuming each symbol has only one order for simplicity.
            bought_price = self.bought_orders[symbol][0]['price']
        else:
            print(f"No orders found for {symbol}.")
            bought_price = None

         # Calculate percentage_change only if bought_price is not zero
        if bought_price is not None:
            percentage_change = ((current_price - bought_price) / bought_price) * 100
            specific_increase = self.specific_percentage_increases.get(symbol, self.default_percentage_increase)            
            
            if percentage_change >= specific_increase:
                print(f"--------{symbol} {percentage_change:.5f}% increase from bought price.--------")
            else:
                print(f"--------{symbol} {-percentage_change:.5f}% decrease from bought price.--------")           

        # Update the highest price for this symbol if it's higher than the recorded highest price
        self.highest_prices[symbol] = max(self.highest_prices.get(symbol, 0), current_price)
        # Save the state after updating the highest price
        self.save_state()    



    if not actions:
        print("No signals detected.")
    else:
        for symbol, action in actions:
            print(f"Signal detected for {symbol}: {action}")  



# Get user inputs  
symbols = input("Enter symbols separated by comma: ").split(',')
quantities = input("Enter quantities separated by comma: ").split(',')
run_counter = 1

while True:

    print("-----------------------------------------")
    print("RUN", run_counter)
    print("-----------------------------------------")

    quantities = [float(q) for q in quantities]
    #quantities = [float(q) for q in quantities if q.strip() != '']


    bot = TradingBot(symbols, dict(zip(symbols, quantities)))

    
################################################################################

################## """############################################################## 

# #Manually REMOVE order (will remove specific order for a specific symbol using price)  
    # symbol = 'BTCUSDT' 

    # # Get orders for the symbol 
    # orders = bot.bought_orders.get(symbol, [])

    # # Find order by price 
    # order_to_remove = next((order for order in orders if order['price'] == 2251.8399999999997), None)

    # # Remove the order if found
    # if order_to_remove:
      # orders.remove(order_to_remove)
      
      # # Update bot.bought_orders
      # bot.bought_orders[symbol] = orders
      
      # print(f"Removed order for {symbol} at price {order_to_remove['price']}")

    # else:
      # print(f"No order found at that price for {symbol}")     
    
################################################################################            
    # Print bought orders for a all symbol 
    for symbol in bot.bought_orders:
      if bot.bought_orders.get(symbol):
         print(f"in dictionary bought orders for {symbol}: {bot.bought_orders[symbol]}")
      else:
         print(f"In dictionary no orders found for {symbol}.")            
            
################################################################################
    # #Reset the data if needed (will delete all orders for all symbols)
    # bot.bought_orders = {}  # Resetting bought orders
    # bot.highest_prices = {}  # Resetting highest prices
    # bot.save_state()  # Saving the reset state

############################################################################
    # #Resetting a specific symbol (will delete all order for a specific symbol)
    # symbol_to_reset = 'ETHUSDT'  # replace with the symbol you want to reset
        
     # # Resetting and deleting bought orders for a specific symbol
    # if symbol_to_reset in bot.bought_orders:
        # del bot.bought_orders[symbol_to_reset]

    # # Resetting and deleting highest prices for a specific symbol
    # if symbol_to_reset in bot.highest_prices:
        # del bot.highest_prices[symbol_to_reset]       
##############################################################################3   


##############################################################################3   

    try:
        bot.run()
    except requests.exceptions.ReadTimeout:
        print("Timeout error with Binance API. Retrying...")
        # Optionally, you can implement a retry mechanism here or just continue to the next iteration.   
    run_counter += 1

            
    # Restart after interval
    #time.sleep(300) # 5 MINS
    time.sleep(120) # 5 MINS



    """   
    # Manually add orders for ETHUSDT
    bot.bought_orders.setdefault('ETHUSDT', []).extend([
        {'price': 3518.0, 'quantity': 0.1},
        {'price': 3727.72, 'quantity': 0.1},
        {'price': 3727.72, 'quantity': 0.1} 
    ])

    # Manually add orders for XRPUSDT
    bot.bought_orders.setdefault('XRPUSDT', []).extend([
        {'price': 0.6063000000000001, 'quantity': 100.0},
        {'price': 0.6455, 'quantity': 100.0},
        {'price': 0.645637, 'quantity': 100.0}
    ])

    # Manually add orders for BTCUSDT
    bot.bought_orders.setdefault('BTCUSDT', []).extend([
        {'price': 66660.992, 'quantity': 0.01},
        {'price': 67654.56661, 'quantity': 0.01},
        {'price': 67665.24, 'quantity': 0.01}
    ])

    # Manually add orders for SOLUSDT
    bot.bought_orders.setdefault('SOLUSDT', []).extend([
        {'price': 174.9975142857143, 'quantity': 7.0},
        {'price': 192.17, 'quantity': 7.0},
        {'price': 192.33002857142856, 'quantity': 7.0}
    ])
    print(bot.bought_orders) """
