# Time Series Sales Analytics Chatbot

An intelligent AI-powered chatbot for analyzing time series sales data with forecasting capabilities, built using LangChain and Claude Sonnet 4.6.

![Architecture Diagram](./data/time_series_design.png)

## 🌟 Overview

This project implements an agentic AI system that enables natural language queries over sales time series data. The chatbot understands user questions, filters data by dimensions (city, shop, brand, container), and performs various analytical operations with temporal filtering.

## 🏗️ Architecture

The system follows a **separation of concerns** design pattern:

- **Dimension Filtering**: The `filter_the_data` tool handles categorical dimensions (city, shop, brand, container)
- **Temporal Filtering**: Individual analysis tools handle date filtering (start_date, end_date parameters)
- **Global Cache**: Filtered data is stored in a global cache for reuse across multiple analysis operations
- **Multi-layer Security**: Input and output guardrails protect against malicious inputs and sensitive data leakage

### Main Flow

```
User Query → Input Guardrails → AI Agent → Dimension Filter → Analysis Tools → Output Guardrails → Response
```

## 🔧 Features

### Analysis Tools

1. **get_sales_data**: Retrieve historical sales data for a specific date range
2. **get_sales_forecast**: Get forecast predictions for future periods
3. **calculate_sales_growth**: Calculate growth rates (QoQ, YoY, MoM) for specified periods
4. **calculate_forecast_accuracy**: Measure forecast accuracy using various metrics (MAPE, RMSE, MAE)
5. **plot_the_data**: Generate visualizations (line charts, bar charts, etc.)
6. **create_report**: Generate comprehensive analytical reports

### Security Guardrails

**Input Validation:**
- SQL Injection Prevention
- Malicious Code Detection
- Input Sanitization
- Rate Limiting

**Output Validation:**
- PII Detection & Removal
- Data Validation
- Error Handling
- Format Sanitization

## 📊 Data Structure

The system works with time series data containing:

- **Dimensions**: city, shop, brand, container, timeseries_id
- **Temporal**: date (monthly granularity)
- **Metrics**: sales (historical), yhat (forecasted values)

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Dependencies

- `langchain` - For agent orchestration
- `langchain-openrouter` - OpenRouter integration for Claude models
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `python-dotenv` - Environment variable management

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. Prepare your data:
   - Place training data in `data/train.csv`
   - Place forecast data in `data/forecast_results.csv`

### Usage

Run the agent in `agent_experiments.ipynb`:

```python
from langchain.agents import create_agent
from sales_agent_tools import *

# Create the agent
agent = create_agent(
    model="openrouter:anthropic/claude-sonnet-4-6",
    tools=[get_sales_data, get_sales_forecast, calculate_sales_growth, 
           calculate_forecast_accuracy, filter_the_data, plot_the_data, create_report],
    system_prompt=sys_prompt
)

# Query the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What are the sales figures for Q1 2018 for shop_1 in Athens?"}]
})
```

## 💡 Example Queries

- "What are the sales figures for Q1 2018 for shop_1 and city Athens?"
- "Show me the QoQ sales growth rates for Q1 2017 for shop_1 in Athens"
- "What's the forecast accuracy for Athens stores in 2017?"
- "Plot the sales trends for brand X in Paris over the last 6 months"
- "Create a report summarizing sales performance for all shops in London"

## 📁 Project Structure

```
agentic_ai/
├── agent_experiments.ipynb          # Main experimentation notebook
├── sales_agent_tools.py             # Core analysis tools implementation
├── tools.py                          # Utility functions
├── html_display_utils.py            # HTML formatting for outputs
├── design.drawio                     # Architecture flowchart (Draw.io)
├── requirements.txt                  # Python dependencies
├── data/
│   ├── train.csv                    # Historical sales data
│   ├── forecast_results.csv         # Forecast predictions
│   ├── test.csv                     # Test data
│   └── time_series_design.png       # Architecture diagram
└── tmp/
    └── credentials.json             # API credentials (gitignored)
```

## 🎯 Key Design Principles

### 1. Separation of Concerns
- **Dimensions** are filtered once using `filter_the_data`
- **Dates** are filtered per-tool for flexible temporal analysis
- Each tool has a single, well-defined responsibility

### 2. Global Cache Pattern
- `_filtered_data` variable stores dimensionally filtered dataset
- Prevents redundant filtering operations
- Improves performance for multi-step analysis

### 3. Date-Level Aggregation
- Sales data is aggregated to daily level across all dimensions
- Ensures consistent time series analysis
- Handles multiple granularities (daily, monthly, quarterly, yearly)

## 🔒 Security Considerations

The chatbot implements multiple security layers:

1. **Input Sanitization**: Validates and cleans all user inputs
2. **Query Validation**: Prevents SQL injection and malicious code execution
3. **Rate Limiting**: Protects against abuse and excessive API calls
4. **Output Filtering**: Removes PII and sensitive information from responses
5. **Error Handling**: Gracefully handles exceptions without exposing system details

## 🛠️ Development

### Adding New Tools

1. Define the tool function in `sales_agent_tools.py`
2. Add appropriate docstring for the agent to understand tool usage
3. Register the tool in the agent creation call
4. Update the system prompt to include the new tool

### Customizing the Agent

Modify the `sys_prompt` variable in `agent_experiments.ipynb` to:
- Change the agent's personality
- Add domain-specific knowledge
- Define custom workflows
- Set response formatting preferences

## 📈 Performance Optimization

- **Caching**: Filtered data is cached globally to avoid redundant operations
- **Lazy Loading**: Data is loaded only when needed
- **Vectorized Operations**: Uses pandas vectorization for fast computations
- **Efficient Aggregation**: Date-level aggregation minimizes memory usage

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Claude Sonnet 4.6](https://www.anthropic.com/claude) via [OpenRouter](https://openrouter.ai/)
- Data visualization with [Matplotlib](https://matplotlib.org/)
- Data processing with [Pandas](https://pandas.pydata.org/)

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is an experimental AI agent system. Always validate results and apply appropriate domain expertise when making business decisions based on the analysis.
