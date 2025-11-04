# SoilWise 🌱

SoilWise is an intelligent agricultural solution that combines soil analysis, crop recommendations, and smart irrigation systems to help farmers make data-driven decisions for optimal crop yield. Powered by advanced IoT sensors and Machine Learning algorithms, SoilWise analyzes your soil's type, moisture levels, pH, and nutrient content.

## Features

- **Soil Analysis**: Advanced soil type detection and analysis using computer vision
- **AgriShield**: Crop disease detection and treatment recommendations
- **IrrigAIte**: Smart irrigation management system
- **Crop Yield Prediction**: ML-powered crop yield forecasting
- **Agricultural Chatbot**: Interactive assistance for farming queries
- **Research Fact Checker**: Validate agricultural research findings

## Project Structure

```plaintext
echofarm/
├── app.py                 # Main Streamlit application
├── assets/               # Image assets for soil and crop types
├── mcp/                 # Model Context Protocol implementation
├── model/               # ML models and inference files
├── notebooks/          # Research and analysis notebooks
├── pages/              # Streamlit application pages
└── src/                # Data sources and datasets
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/antonie-riziki/SoilWise.git
    cd SoilWise
    ```

2. Create and activate a virtual environment (recommended):

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Linux/MacOS
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Start the MCP server:

    ```bash
    cd echofarm/mcp
    python server_example.py
    ```

2. In a new terminal, run the Streamlit application:

    ```bash
    cd echofarm
    streamlit run app.py
    ```

The application will be available at `http://localhost:8501` by default.

## Components

### 1. Soil Analysis

Upload parameters of soil samples for automatic soil type classification and detailed analysis of soil properties.

### 2. AgriShield

Detect and diagnose crop diseases through image analysis and get recommended treatments.

### 3. IrrigAIte

Smart irrigation system that provides water management recommendations based on soil conditions and weather data.

### 4. Crop Yield Prediction

Machine learning models to predict crop yields based on historical data and current conditions.

### 5. Agricultural Chatbot

Interactive AI assistant for farming-related queries and guidance.

### 6. Research Fact Checker

Validate agricultural research findings and get evidence-based information.

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```plaintext
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework
- [OpenAI](https://openai.com/) for AI capabilities
- [Google AI](https://ai.google/) for generative AI features
- [Africa's Talking](https://africastalking.com/) for communication services

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
AT_API_KEY=your_africastalking_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework
- [OpenAI](https://openai.com/) for AI capabilities
- [Google AI](https://ai.google/) for generative AI features
- [Africa's Talking](https://africastalking.com/) for communication services
- 

