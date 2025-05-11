# Road Accident Analysis System


## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [File Structure](#file-structure)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Kenya Road Accident Analysis System is a comprehensive web application designed to process, analyze, and visualize road accident data across Kenya. This system helps transportation authorities, researchers, and policymakers identify accident patterns, high-risk areas, and contributing factors to improve road safety.

## Features

- **Data Processing**:
  - Handles both CSV and Excel files
  - Automatic data validation and cleaning
  - Flexible column name matching

- **Analytical Capabilities**:
  - Temporal analysis (yearly, monthly, daily, hourly trends)
  - Spatial analysis (by county and road type)
  - Accident hotspot detection using clustering
  - Severity prediction with machine learning

- **Visualization**:
  - Interactive maps with Folium
  - Statistical charts and graphs
  - Feature importance visualization

- **User Interface**:
  - Responsive web interface
  - County-based filtering
  - Comprehensive reporting
  - Error handling and user feedback

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Warrenchris/road-accident-analyzer.git
   cd road-accident-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Access the application at:
   ```
   http://localhost:5000
   ```

## Usage

### Data Upload
1. Navigate to the Upload page
2. Select a CSV or Excel file with accident data
3. The system will automatically process and validate the data

### Running Analyses
1. From the dashboard, select an analysis type:
   - Temporal Patterns
   - Spatial Distribution
   - Accident Hotspots
   - Severity Prediction
2. Apply county filters if needed
3. View and interpret the results

### Sample Data
A sample dataset is provided in `data/sample_data.csv` for testing purposes.

## Data Requirements

The system expects data with the following columns (exact names can vary):

| Required Column | Description | Example |
|-----------------|-------------|---------|
| Date | Accident date (DD/MM/YYYY) | 15/03/2023 |
| Time | Accident time (24-hour format) | 14:30 |
| County | County where accident occurred | Nairobi |
| Latitude | GPS coordinate (optional) | -1.286389 |
| Longitude | GPS coordinate (optional) | 36.817223 |
| Casualties | Number of casualties | 2 |
| Vehicles Involved | Number of vehicles | 3 |
| Road Type | Type of road | Highway |
| Weather | Weather conditions | Rainy |

## File Structure

```
kenya-accident-app/
├── app.py                  # Main application
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── data/
│   ├── sample_data.csv     # Example data
│   └── uploads/            # User-uploaded files
├── models/
│   └── accident_analyzer.py # Analysis logic
├── static/
│   ├── css/
│   │   └── styles.css      # Custom styles
│   ├── js/
│   │   └── scripts.js      # Custom JavaScript
│   └── images/             # Static images
└── templates/
    ├── base.html           # Base template
    ├── index.html          # Dashboard
    ├── upload.html         # Data upload
    ├── analysis.html       # Analysis results
    ├── map.html            # Map visualization
    ├── report.html         # Detailed reports
    └── error.html          # Error pages
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/upload` | GET, POST | Data upload page |
| `/analyze` | POST | Run analysis |
| `/analyze/temporal` | GET | Temporal analysis results |
| `/analyze/spatial` | GET | Spatial analysis results |
| `/analyze/hotspots` | GET | Hotspot analysis results |
| `/analyze/severity` | GET | Severity prediction results |

## Contributing

We welcome contributions to improve this system. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kenya National Transport and Safety Authority (NTSA)
- Kenya National Bureau of Statistics (KNBS)
- OpenStreetMap for geospatial data

---

For any questions or support, please contact warrenchris745@gmail.com.
