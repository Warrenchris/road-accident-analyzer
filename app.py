import matplotlib
matplotlib.use('Agg')  # Must be before other matplotlib imports
from flask import Flask, render_template, request, jsonify, send_from_directory
from models.accident_analyzer import KenyaAccidentAnalyzer
import pandas as pd
import numpy as np
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from io import BytesIO
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Initialize analyzer
analyzer = KenyaAccidentAnalyzer()

# County coordinates mapping for Kenya
COUNTY_COORDINATES = {
    'NAIROBI': (-1.286389, 36.817223),  # Nairobi City
    'MOMBASA': (-4.0435, 39.6682),     # Mombasa
    'KWALE': (-4.1737, 39.4521),
    'KILIFI': (-3.5107, 39.9093),
    'TANA RIVER': (-1.6519, 39.6516),
    'LAMU': (-2.2696, 40.9006),
    'TAITA TAVETA': (-3.3163, 38.4841),
    'GARISSA': (-0.4536, 39.6461),
    'WAJIR': (1.7500, 40.0667),
    'MANDERA': (3.9333, 41.8667),
    'MARSABIT': (2.3346, 37.9909),
    'ISIOLO': (0.3556, 37.5833),
    'MERU': (0.0500, 37.6500),
    'THARAKA-NITHI': (-0.2969, 37.9043),
    'EMBU': (-0.5375, 37.4506),
    'KITUI': (-1.3670, 38.0106),
    'MACHAKOS': (-1.5167, 37.2667),
    'MAKUENI': (-2.2833, 37.8333),
    'NYANDARUA': (-0.5333, 36.5833),
    'NYERI': (-0.4167, 36.9500),
    'KIRINYAGA': (-0.5000, 37.2833),
    "MURANG'A": (-0.7833, 37.0333),
    'KIAMBU': (-1.0167, 36.8667),
    'TURKANA': (3.1167, 35.6000),
    'WEST POKOT': (1.4167, 35.1667),
    'SAMBURU': (1.1167, 36.6833),
    'TRANS NZOIA': (1.0167, 34.9833),
    'UASIN GISHU': (0.5167, 35.2833),
    'ELGEYO-MARAKWET': (0.5167, 35.5167),
    'NANDI': (0.2000, 35.1000),
    'BARINGO': (0.4667, 35.9667),
    'LAIKIPIA': (0.1667, 36.8333),
    'NAKURU': (-0.3031, 36.0800),
    'NAROK': (-1.0833, 35.8667),
    'KAJIADO': (-1.8500, 36.7833),
    'KERICHO': (-0.3667, 35.2833),
    'BOMET': (-0.7833, 35.3333),
    'KAKAMEGA': (0.2833, 34.7500),
    'VIHIGA': (0.0833, 34.7167),
    'BUNGOMA': (0.5667, 34.5667),
    'BUSIA': (0.4667, 34.1000),
    'SIAYA': (0.0667, 34.2833),
    'KISUMU': (-0.1022, 34.7617),
    'HOMA BAY': (-0.5167, 34.4500),
    'MIGORI': (-1.0667, 34.4667),
    'KISII': (-0.6833, 34.7667),
    'NYAMIRA': (-0.5667, 34.9333)
}
@app.route('/')
def index():
    """Render the home page"""
    stats = analyzer.get_summary_stats() if analyzer.analyzer else None
    current_year = datetime.now().year
    return render_template('index.html', stats=stats, current_year=current_year)

def load_and_preprocess_data(filepath):
    """Load and preprocess the accident data from Excel/CSV with flexible column handling"""
    try:
        # Read the file
        if filepath.endswith('.csv'):
            raw_data = pd.read_csv(filepath)
        else:
            raw_data = pd.read_excel(filepath, engine='openpyxl')
        
        print(f"File loaded successfully. Found columns: {raw_data.columns.tolist()}")
        
        # Create a basic structure if the file doesn't have all required columns
        processed_data = pd.DataFrame()
        
        # Handle date - look for any column that might contain date information
        date_columns = [col for col in raw_data.columns if 'DATE' in str(col).upper() or 'DAY' in str(col).upper()]
        if date_columns:
            # Try the first column that looks like a date
            try:
                processed_data['date'] = pd.to_datetime(raw_data[date_columns[0]], errors='coerce')
                print(f"Using {date_columns[0]} as date column")
            except:
                print(f"Could not convert {date_columns[0]} to date, creating default dates")
                processed_data['date'] = pd.Timestamp('today')
        else:
            # Create a default date if none found
            print("No date column found, creating default dates")
            processed_data['date'] = pd.Timestamp('today')
        
        # Extract time information if available
        time_columns = [col for col in raw_data.columns if 'TIME' in str(col).upper() or 'HOUR' in str(col).upper()]
        if time_columns:
            processed_data['time'] = raw_data[time_columns[0]].astype(str)
            print(f"Using {time_columns[0]} as time column")
        else:
            processed_data['time'] = '12:00'  # Default to noon
            print("No time column found, using default time")
        
        # Map location data - counties
        location_columns = [col for col in raw_data.columns if 'COUNTY' in str(col).upper() or 'LOCATION' in str(col).upper()]
        if location_columns:
            processed_data['county'] = raw_data[location_columns[0]].astype(str).str.upper()
            print(f"Using {location_columns[0]} as county column")
        else:
            # Try to use any column that might contain location information
            if 'PLACE' in raw_data.columns:
                processed_data['county'] = raw_data['PLACE'].astype(str).str.upper()
                print("Using PLACE as county column")
            else:
                processed_data['county'] = 'UNKNOWN'
                print("No county column found, using default")
        
        # Other basic columns
        # Road information
        road_columns = [col for col in raw_data.columns if 'ROAD' in str(col).upper() or 'HIGHWAY' in str(col).upper()]
        if road_columns:
            processed_data['road'] = raw_data[road_columns[0]]
            print(f"Using {road_columns[0]} as road column")
        else:
            processed_data['road'] = 'UNKNOWN'
            print("No road column found, using default")
            
        # Vehicle count
        vehicle_columns = [col for col in raw_data.columns if 'VEHICLE' in str(col).upper() or 'MV' in str(col).upper()]
        if vehicle_columns:
            processed_data['vehicles_involved'] = pd.to_numeric(raw_data[vehicle_columns[0]], errors='coerce').fillna(1).astype(int)
            print(f"Using {vehicle_columns[0]} as vehicles column")
        else:
            processed_data['vehicles_involved'] = 1
            print("No vehicles column found, using default")
            
        # Casualties
        casualty_columns = [col for col in raw_data.columns if 'NO.' in str(col) or 'CASUALT' in str(col).upper() or 'INJUR' in str(col).upper()]
        if casualty_columns:
            processed_data['casualties'] = pd.to_numeric(raw_data[casualty_columns[0]], errors='coerce').fillna(0).astype(int)
            print(f"Using {casualty_columns[0]} as casualties column")
        else:
            processed_data['casualties'] = 0
            print("No casualties column found, using default")
        
        # Try to get place info
        if 'PLACE' in raw_data.columns:
            processed_data['place'] = raw_data['PLACE']
        else:
            processed_data['place'] = 'UNKNOWN'
        
        # Cause information
        cause_columns = [col for col in raw_data.columns if 'CAUSE' in str(col).upper() or 'REASON' in str(col).upper()]
        if cause_columns:
            processed_data['cause_code'] = raw_data[cause_columns[0]]
            print(f"Using {cause_columns[0]} as cause column")
        else:
            processed_data['cause_code'] = 'UNKNOWN'
            print("No cause column found, using default")
        
        # Add coordinates based on county
        from app import COUNTY_COORDINATES  # Import from your main app
        processed_data['latitude'] = processed_data['county'].map(
            lambda x: COUNTY_COORDINATES.get(str(x).upper().strip(), (np.nan, np.nan))[0]
        )
        processed_data['longitude'] = processed_data['county'].map(
            lambda x: COUNTY_COORDINATES.get(str(x).upper().strip(), (np.nan, np.nan))[1]
        )
        
        # Set default weather (not in original data)
        processed_data['weather'] = 'UNKNOWN'
        
        print(f"Data preprocessing completed successfully. Shape: {processed_data.shape}")
        return processed_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with better error reporting"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            processed_data = load_and_preprocess_data(filepath)
            
            if processed_data is not None:
                analyzer.load_and_preprocess_data(processed_data)
                return jsonify({
                    'success': 'File processed successfully',
                    'stats': analyzer.get_summary_stats(),
                    'columns_found': list(processed_data.columns)
                })
            else:
                return jsonify({
                    'error': 'Could not process file',
                    'details': 'Check if the file has the required columns'
                }), 400
                
        except Exception as e:
            return jsonify({
                'error': 'File processing failed',
                'details': str(e)
            }), 500
    
    return jsonify({
        'error': 'Invalid file type',
        'allowed_types': app.config['ALLOWED_EXTENSIONS']
    }), 400

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Perform analysis based on user parameters"""
    try:
        analysis_type = request.form.get('analysis_type', 'temporal')
        
        if analysis_type == 'temporal':
            results = analyzer.analyzer.temporal_analysis()
            return render_template('analysis.html', results=results, analysis_type='temporal')
        
        elif analysis_type == 'spatial':
            results = analyzer.analyzer.spatial_analysis()
            return render_template('analysis.html', results=results, analysis_type='spatial')
        
        elif analysis_type == 'hotspots':
            # Get valid coordinates (non-null)
            valid_coords = analyzer.analyzer.data.dropna(subset=['latitude', 'longitude'])
            n_samples = len(valid_coords)
            
            # Dynamically adjust clusters based on available data
            n_clusters = min(5, max(2, n_samples - 1))  # At least 2 clusters, max 5
            
            if n_samples < 2:
                return render_template('error.html', 
                                    error="Need at least 2 locations for hotspot analysis")
            
            kmeans, scaler = analyzer.analyzer.cluster_accident_hotspots(n_clusters=n_clusters)
            return render_template('map.html', 
                                map_data=analyzer.visualizer.plot_cluster_hotspots(kmeans, scaler)._repr_html_())
        
       
        
        elif analysis_type == 'severity':
            model, report = analyzer.analyzer.predict_severity()
            return render_template('report.html', report=report)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/analyze/severity', methods=['GET', 'POST'])
def analyze_severity():
    if not hasattr(analyzer, 'analyzer') or analyzer.analyzer is None:
        return render_template('error.html', error="No data loaded for analysis")

    result = analyzer.analyzer.predict_severity()

    if result['status'] == 'error':
        return render_template('error.html', error=result['message'])

    return render_template('severity_report.html',
                         report=result['report'],
                         features=result['features'],
                         sample_size=result['sample_size'])

@app.route('/map')
def show_map():
    """Display the interactive accident map"""
    try:
        kenya_map = analyzer.visualizer.plot_kenya_map()
        return render_template('map.html', map_data=kenya_map._repr_html_())
    except Exception as e:
        return render_template('error.html', error=str(e))

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Try to load sample data if exists
    sample_path = os.path.join('data', 'accidents.csv')
    if os.path.exists(sample_path):
        sample_data = load_and_preprocess_data(sample_path)
        if sample_data is not None:
            analyzer.load_and_preprocess_data(sample_data)
    
    app.run(debug=True)