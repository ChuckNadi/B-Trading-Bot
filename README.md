# BTB9: Automated Cryptocurrency Trading Bot

BTB9 is an advanced automated trading bot designed to trade cryptocurrencies on the Binance exchange. It leverages technical indicators such as EMA, RSI, MACD, and stochastic RSI to analyze market conditions and execute trades accordingly. The bot supports trailing stop-loss mechanisms, manual order adjustments, and compound trading strategies.

---

## Key Features
- **Dynamic Market Analysis**: Uses EMA, RSI, MACD, and stochastic RSI for trade signals.
- **Automated Trading**: Executes buy/sell orders based on market trends and custom conditions.
- **Risk Management**: Implements trailing stop-loss and customizable sell conditions to minimize losses.
- **Customizable Configurations**: User-defined trading pairs, quantities, and stop-loss thresholds.
- **Performance Tracking**: Logs trades, visualizes profits/losses, and tracks wallet balance.
- **Manual Order Initialization**: Supports manual input of pre-existing orders.

---

## Demo
![Trading Bot Screenshot](./assets/demo.png)

---

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/BTB9.git
cd BTB9
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Binance API Key Setup
1. Copy the config template:
```bash
cp configs/config_template.json configs/config.json
```
2. Add your Binance API key and secret to `configs/config.json`:
```json
{
  "api_key": "YOUR_BINANCE_API_KEY",
  "api_secret": "YOUR_BINANCE_API_SECRET"
}
```
3. Alternatively, export API keys as environment variables:
```bash
export BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
export BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"
```

---

## Usage
### Run the Trading Bot
```bash
python src/trading_bot.py
```
Modify symbols and quantities directly in the bot initialization or via the command line.

---

## Configuration
The bot can be customized via `config.json` or directly by modifying `src/trading_bot.py`.

### Sample Configuration (Manual Orders Initialization)
```python
manual_orders = {
  'BTCUSDT': [{'price': 66660.99, 'quantity': 0.01}],
  'ETHUSDT': [{'price': 3518.0, 'quantity': 0.1}]
}
```

---

## Project Structure
```
BTB9/
│
├── data/                     # Historical data or sample data
│   └── sample_data.csv
│
├── notebooks/                # Jupyter Notebooks for analysis or visualization
│   └── trade_analysis.ipynb
│
├── src/                      # Core trading bot scripts
│   ├── trading_bot.py
│   └── utils.py
│
├── tests/                    # Unit tests
│   └── test_trading_bot.py
│
├── configs/                  # Config files
│   └── config.json
│
├── requirements.txt          # Required Python packages
├── README.md                 # Project description
├── LICENSE                   # License file
└── .gitignore                # Ignore unnecessary files
```

---

## Visualization and Trade Analysis
- **Jupyter Notebooks** in `notebooks/` visualize trade performance.
- Example:
```python
plt.plot(df['time'], df['close'], label='BTCUSDT Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BTCUSDT Price Over Time')
plt.show()
```

---

## Sample Trades Visualization
![Trade Analysis](./assets/trade_plot.png)

---

## Roadmap
- [ ] Implement LSTM-based price prediction
- [ ] Develop a real-time dashboard for trade monitoring
- [ ] Email notifications for trade execution
- [ ] Backtesting module for evaluating strategies

---

## Contributing
Contributions are welcome! Feel free to fork the project, submit issues, or create pull requests.

1. Fork the repository.
2. Create a new branch for your feature:
```bash
git checkout -b feature-new-trade-logic
```
3. Commit changes and push to your branch.
4. Submit a pull request.

---

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). See `LICENSE` for more details.

---



