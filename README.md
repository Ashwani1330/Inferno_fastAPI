# Inferno VR Fire-Safety Training API

![Inferno VR](https://img.shields.io/badge/Inferno-VR-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

A comprehensive API for the Immersive Navigation for Fire Emergency Response & Neutralization Operations (Inferno) research project. This API processes performance data from VR fire safety training simulations and provides analytics and visualization tools.

## 🔥 Overview

Inferno VR is a research-driven virtual reality training platform designed to improve fire emergency response effectiveness. The API collects performance metrics from VR training sessions and provides data analytics, personalized performance reports, and a researcher dashboard.

> Dashboard Link: [Inferno-dashboard](https://inferno-fastapi.onrender.com/api/dashboard)

## 🚀 Features

- **Performance Data Collection**: Record and store user performance metrics in fire emergency simulations
- **Performance Scoring**: Calculate evacuation efficiency scores based on custom algorithms
- **Personalized Reports**: Generate and email personalized performance reports to participants
- **Analytics Dashboard**: Interactive visualization dashboard for researchers
- **Data Analysis**: Advanced statistical analysis of performance data including:
  - Demographic analysis
  - Scene type comparisons
  - Time-series trend analysis
  - Correlations between parameters
  - Predictive modeling

## 📊 Dashboard

The analytics dashboard provides researchers with comprehensive visualization tools:

- Key performance metrics
- Demographic breakdowns
- Performance comparisons by age group, scene type, and difficulty
- Correlation analysis
- Time trend analysis
- Task-specific performance metrics

## 🧩 Architecture

The API is built with FastAPI and follows a modular architecture:

- **Routes**: API endpoints for performance data, analytics, and dashboard
- **Models**: Pydantic models for data validation
- **Services**: Business logic services for MongoDB, email, and analysis
- **Utils**: Helper functions and score calculation algorithms

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/inferno-fastapi.git
cd inferno-fastapi
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables in .env file:
```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=inferno
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_USERNAME=your-email@example.com
EMAIL_PASSWORD=your-password
EMAIL_FROM=noreply@inferno-vr.com
REDIS_URL=redis://localhost:6379
```

4. Run the application:
```bash
python run.py
```

The API will be available at `http://localhost:10000`.

## 💻 API Endpoints

### Performance Endpoints
- `POST /api/performance` - Submit performance data from VR sessions
- `GET /api/performance` - Retrieve all performance records
- `GET /api/performance/report` - Generate HTML performance report
- `GET /api/performance/analysis` - Get detailed statistical analysis

### Analytics
- `GET /analytics` - Redirects to dashboard
- `GET /api/dashboard` - Interactive analytics dashboard
- `GET /api/dashboard-tab/{tab_name}` - Load specific dashboard tab data

## 📋 Data Model

Performance data includes:
- User information (email, age)
- Session information (scene type, difficulty)
- Time metrics (time to find extinguisher, time to extinguish fire, etc.)
- Performance score
- Timestamp

## 🔌 Dependencies

- FastAPI - Modern Python web framework
- Motor - Asynchronous MongoDB driver
- Pandas/NumPy - Data analysis
- Matplotlib/Seaborn - Data visualization
- scikit-learn - Machine learning for predictive modeling
- Redis/FastAPI-cache - Caching for performance optimization

## 🧠 Performance Score Calculation

The application uses a custom algorithm to calculate an Evacuation Efficiency Score (EES) based on:
- User age (weighted differently for different age groups)
- Time to find extinguisher
- Time to extinguish fire
- Time to trigger alarm
- Time to find exit
- Mobility score

## 🔒 Security

- Anonymous data collection option
- Data anonymization for research dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
