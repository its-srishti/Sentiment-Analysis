# Sentiment-Analysis

# Management Sentiment Analysis

> "The market is a voting machine in the short run, but a weighing machine in the long run." - Benjamin Graham

**A comprehensive Python framework for analyzing management sentiment in financial disclosures and its relationship to stock market performance.**

This project quantifies the tone of corporate communications using the Loughran-MacDonald financial dictionary, focusing on Management Discussion & Analysis (MD&A) sections from 10-K filings. By analyzing sentiment trends and changes in management language, researchers and investors can gain insights into corporate outlook and identify potential investment opportunities based on management's expressed optimism or caution.

## Overview

Sentiment analysis in finance goes beyond traditional natural language processing by addressing the unique vocabulary and context of financial disclosures. This project implements dictionary-based sentiment analysis specifically designed for financial documents, utilizing the SEC's EDGAR database and the specialized Loughran-MacDonald dictionary to extract meaningful sentiment signals from corporate filings.

## Features

### Core Capabilities
- **Financial Text Processing**: Specialized sentiment analysis for financial documents
- **SEC EDGAR Integration**: Direct access to 10-K filings and MD&A sections
- **Loughran-MacDonald Dictionary**: Finance-specific sentiment word classification
- **Historical Analysis**: Track sentiment changes over time (1993-present)
- **Investment Strategy Testing**: Evaluate sentiment-based portfolio performance

### Key Analyses
- **Sentiment Scoring**: Net sentiment calculation using positive/negative word counts
- **Temporal Analysis**: Year-over-year sentiment changes
- **Market Correlation**: Relationship between sentiment and stock returns
- **Portfolio Construction**: Decile-based investment strategies
- **Performance Attribution**: Same-year vs. forward-looking return analysis

## Installation

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn tqdm
```

### Required Libraries
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Visualization
- `scikit-learn` - Text vectorization and feature extraction
- `tqdm` - Progress tracking
- `finds` - Custom financial data library (included)

### Database Setup
The project requires access to financial databases:
```python
# Configure database credentials in secret.py
credentials = {
    'sql': {'host': 'your_host', 'user': 'your_user', 'password': 'your_password'},
    'redis': {'host': 'redis_host', 'port': 6379},
    'fred': {'api_key': 'your_fred_api_key'}
}
```

## Quick Start

### Basic Sentiment Analysis

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load Loughran-MacDonald dictionary
source = "https://drive.google.com/uc?export=download&id=1ptUgGVeeUGhCbaKL14Ri3Xi5xOKkPkUD"
df = pd.read_csv(source, sep=',')

# Extract sentiment words
words = {
    'positive': set(df.loc[df['Positive'] != 0, 'Word'].str.lower()),
    'negative': set(df.loc[df['Negative'] != 0, 'Word'].str.lower())
}

print(f"Positive words: {len(words['positive'])}")
print(f"Negative words: {len(words['negative'])}")
```

### Processing 10-K MD&A Sections

```python
from finds.unstructured import Edgar
from finds.database import SQL
from finds.structured import CRSP, BusDay

# Initialize data sources
ed = Edgar(paths['10X'], zipped=True)
crsp = CRSP(sql=sql, bd=bd, rdb=rdb)

# Load MD&A documents
rows = pd.DataFrame(ed.open(form='10-K', item='mda10K'))
print(f"Total documents: {len(rows)}")
print(f"Date range: {min(rows['date'])} to {max(rows['date'])}")
```

### Sentiment Calculation

```python
# Set up vectorizers
vectorizer = CountVectorizer(
    strip_accents='unicode',
    lowercase=True,
    stop_words=generic_stopwords,
    token_pattern=r"\b[^\d\W][^\d\W][^\d\W]+\b"
)

sentiment_vectorizer = CountVectorizer(
    strip_accents='unicode',
    lowercase=True,
    token_pattern=r"\b[^\d\W][^\d\W][^\d\W]+\b"
)

# Fit on sentiment dictionary
sentiment_vectorizer.fit([" ".join(words['positive'].union(words['negative']))])

# Calculate sentiment scores
features = pd.Series(sentiment_vectorizer.get_feature_names_out())
sentiment_points = (features.isin(words['positive']).astype(int).values 
                   - features.isin(words['negative']).astype(int).values)
```

## Project Structure

```
sentiment-analysis/
├── README.md
├── requirements.txt
├── secret.py                    # Database credentials (not in repo)
├── notebooks/
│   └── sentiment_analysis.ipynb
├── finds/
│   ├── database.py             # SQL and Redis connections
│   ├── structured.py           # CRSP data access
│   ├── unstructured.py         # Edgar filing access
│   ├── readers.py              # Data readers
│   ├── recipes.py              # Analysis utilities
│   └── utils.py                # General utilities
├── data/
│   └── dictionaries/           # Sentiment dictionaries
└── results/
    └── analysis/               # Output files and plots
```

## Key Components

### Loughran-MacDonald Dictionary

The foundation of this analysis uses the finance-specific sentiment dictionary:

- **Positive Words** (354 terms): "outperform", "efficiency", "leadership", "advantageous"
- **Negative Words** (2,355 terms): "threatening", "panic", "contradiction", "uncontrolled"
- **Stop Words**: Generic terms excluded from analysis
- **Domain-Specific**: Designed specifically for financial contexts

### SEC EDGAR Integration

Access to comprehensive filing data:
- **Forms Covered**: 10-K annual reports
- **Text Extraction**: MD&A section identification and parsing
- **Historical Coverage**: 1993-present
- **Company Coverage**: 14,696+ unique companies, 137,691+ documents

### Sentiment Metrics

**Primary Measures:**
- **Net Sentiment**: (Positive words - Negative words) / Total words
- **Sentiment Change**: Year-over-year difference in net sentiment
- **Document Length**: Word count for normalization

## Analysis Results

### Coverage Statistics
- **Time Period**: 1993-2024
- **Companies**: ~15,000 unique firms
- **Documents**: 130,000+ MD&A sections
- **Peak Coverage**: 2008-2010 period

### Filing Patterns
- **Monthly Distribution**: Heavy concentration in March (fiscal year-ends)
- **Day-of-Week**: Tuesday through Thursday peaks
- **Lag Analysis**: Up to 3-month filing delays accounted for

### Performance Analysis

**Investment Strategy Results:**
- **Decile Portfolios**: Top/bottom sentiment deciles
- **Same-Year Returns**: January-December performance
- **Forward Returns**: April-March following year performance
- **Cap-Weighted**: Market capitalization adjustments

## Usage Examples

### Historical Sentiment Trends

```python
# Analyze sentiment trends over time
results = []
for year in range(1999, 2025):
    year_data = data[data['year'] == year]
    median_sentiment = year_data['mdasent'].median()
    results.append({'year': year, 'sentiment': median_sentiment})

trends = pd.DataFrame(results)
trends.plot(x='year', y='sentiment', kind='line')
```

### Portfolio Construction

```python
# Create sentiment-based portfolios
from finds.recipes import fractile_split, weighted_average

def create_sentiment_portfolio(data, year, sentiment_col):
    # Get universe for the year
    universe = data[data['year'] == year].dropna(subset=[sentiment_col])
    
    # Split into deciles
    deciles = fractile_split(universe[sentiment_col], [10, 90])
    
    # Calculate weighted returns
    high_sentiment = weighted_average(
        universe.loc[deciles==1, ['cap', 'ret']], 'cap'
    )['ret']
    
    low_sentiment = weighted_average(
        universe.loc[deciles==3, ['cap', 'ret']], 'cap'
    )['ret']
    
    return high_sentiment - low_sentiment
```

### Economic Correlation

```python
# Compare with economic indicators
from finds.readers import Alfred

alf = Alfred(api_key=credentials['fred']['api_key'])
corporate_profits = alf('CP')  # Corporate Profits series

# Merge with sentiment data for correlation analysis
economic_data = corporate_profits.to_frame().assign(
    year=corporate_profits.index // 10000
).groupby('year').sum()
```

## Research Applications

### Academic Research
- **Behavioral Finance**: Management communication analysis
- **Market Efficiency**: Information content of financial disclosures
- **Corporate Finance**: Management tone and firm performance

### Investment Applications
- **Fundamental Analysis**: Supplement quantitative metrics
- **Risk Management**: Early warning signals from management tone
- **Portfolio Construction**: Sentiment-based factor strategies

### Regulatory Analysis
- **Disclosure Quality**: Evolution of financial communication
- **Market Impact**: Regulatory changes and filing patterns

## Data Sources & Attribution

- **Loughran-MacDonald Dictionary**: Loughran & McDonald (2011)
- **SEC EDGAR**: U.S. Securities and Exchange Commission
- **CRSP Database**: Center for Research in Security Prices
- **FRED Economic Data**: Federal Reserve Bank of St. Louis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new sentiment analysis'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for research and educational purposes only. Sentiment analysis results should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always consult with qualified financial professionals before making investment decisions.

---

*"Sentiment analysis helps quantify the tone of financial disclosures, revealing whether a company's management expresses optimism, caution, or concern."*
