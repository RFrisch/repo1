# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a collection of standalone Python scripts for quantitative finance and financial analysis. Each script is self-contained and can be run independently.

## Scripts

- **Credit scoring Gemini.py**: Credit scoring model using logistic regression with Weight of Evidence (WOE) and Information Value (IV) feature engineering. Generates a points-based scorecard for credit card customer evaluation.
- **duration_convexity.py**: Bond duration and convexity calculator. Outputs results to Excel.
- **signalMA.py**: Moving average crossover trading signal generator with visualization.

## Subrepositories

- **stock-screener/**: Stock screener that identifies stocks trading near their 52-week lows. Outputs results to `data.json` for a frontend (`index.html`) to display. Uses `yfinance` for market data and concurrent futures for parallel fetching. Repo: `RFrisch/stock-screener`.

## Running Scripts

```bash
python "Credit scoring Gemini.py"
python duration_convexity.py
python signalMA.py
python stock-screener/screener.py
```

## Dependencies

Common dependencies across scripts:
- pandas
- numpy
- matplotlib
- scikit-learn (for credit scoring)
- yfinance (for stock screener)
