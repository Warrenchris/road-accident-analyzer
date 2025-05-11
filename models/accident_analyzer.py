import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import base64
from io import BytesIO
from datetime import datetime

class AccidentDataPreprocessor:
    def __init__(self):
        self.kenyan_counties = [
            'Mombasa', 'Kwale', 'Kilifi', 'Tana River', 'Lamu', 
            'Taita Taveta', 'Garissa', 'Wajir', 'Mandera', 
            'Marsabit', 'Isiolo', 'Meru', 'Tharaka-Nithi', 
            'Embu', 'Kitui', 'Machakos', 'Makueni', 'Nyandarua', 
            'Nyeri', 'Kirinyaga', 'Murang\'a', 'Kiambu', 
            'Turkana', 'West Pokot', 'Samburu', 'Trans Nzoia', 
            'Uasin Gishu', 'Elgeyo-Marakwet', 'Nandi', 
            'Baringo', 'Laikipia', 'Nakuru', 'Narok', 'Kajiado', 
            'Kericho', 'Bomet', 'Kakamega', 'Vihiga', 
            'Bungoma', 'Busia', 'Siaya', 'Kisumu', 'Homa Bay', 
            'Migori', 'Kisii', 'Nyamira', 'Nairobi'
        ]
    
    def clean_data(self, df):
        """Clean and standardize accident data"""
        # Handle missing values
        df.fillna({
            'casualties': 0,
            'vehicles_involved': 1,
            'county': 'Unknown',
            'time': '00:00',
            'road_type': 'Unknown',
            'weather': 'Unknown'
        }, inplace=True)
        
        # Standardize county names
        df['county'] = df['county'].apply(
            lambda x: x if x in self.kenyan_counties else 'Unknown'
        )
        
        # Convert date columns
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.day_name()
        
        return df
    
    def feature_engineering(self, df):
        """Create new features from existing data"""
    # Time parsing with explicit formats
        try:
        # Try common time formats
            time_formats = ['%H:%M', '%H:%M:%S', '%H.%M', '%H%M']
            hours = None
            for fmt in time_formats:
                try:
                    hours = pd.to_datetime(df['time'], format=fmt, errors='raise').dt.hour
                    break
                except:
                    continue
        
            if hours is None:
                hours = pd.to_datetime(df['time'], errors='coerce').dt.hour.fillna(12)  # default to noon
    
        except Exception as e:
            print(f"Time parsing error: {e}")
            hours = pd.Series([12] * len(df))  # default to noon if all parsing fails
    
    # Time of day categories
        df['time_of_day'] = pd.cut(
            hours,
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        ).fillna('Afternoon')
    
    
    
    # Severity classification
        df['severity'] = pd.cut(
            df['casualties'],
            bins=[-1, 0, 3, 10, float('inf')],
            labels=['No injury', 'Minor', 'Major', 'Fatal']
        ).fillna('No injury')
    
        return df
    # Rest of the method remains the same

def cluster_accident_hotspots(self, n_clusters=5):
    """Cluster accident locations to identify hotspots"""
    if 'latitude' not in self.data.columns or 'longitude' not in self.data.columns:
        raise ValueError("Latitude and longitude data required for clustering")
        
    coords = self.data[['latitude', 'longitude']].dropna()
    
    # Adjust clusters if not enough data points
    n_clusters = min(n_clusters, len(coords)-1)
    if n_clusters < 1:
        raise ValueError("Not enough locations for clustering")
    
    # Standardize coordinates
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_coords)
    
    # Add cluster labels to data
    self.data.loc[coords.index, 'cluster'] = clusters
    
    return kmeans, scaler

class AccidentVisualizer:
    def __init__(self, data):
        self.data = data
    
    def plot_to_base64(self, plt):
        """Convert matplotlib plot to base64 encoded image"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def plot_temporal_trends(self, temporal_results):
        """Plot temporal analysis results and return as base64"""
        plt.figure(figsize=(15, 10))
        
        # Yearly trend
        plt.subplot(2, 2, 1)
        temporal_results['yearly_trend'].plot(kind='line')
        plt.title('Yearly Accident Trend')
        
        # Monthly pattern
        plt.subplot(2, 2, 2)
        temporal_results['monthly_pattern'].plot(kind='bar')
        plt.title('Monthly Accident Pattern')
        
        # Day of week pattern
        plt.subplot(2, 2, 3)
        temporal_results['day_of_week_pattern'].plot(kind='bar')
        plt.title('Day of Week Accident Pattern')
        
        plt.tight_layout()
        return self.plot_to_base64(plt)
    
    def plot_spatial_distribution(self, spatial_results):
        """Plot spatial analysis results and return as base64"""
        plt.figure(figsize=(15, 5))
        
        # By county
        plt.subplot(1, 2, 1)
        spatial_results['by_county'].head(10).plot(kind='barh')
        plt.title('Top 10 Counties by Accident Count')
        
        # By road type if available
        if 'by_road_type' in spatial_results:
            plt.subplot(1, 2, 2)
            spatial_results['by_road_type'].plot(kind='bar')
            plt.title('Accidents by Road Type')
        
        plt.tight_layout()
        return self.plot_to_base64(plt)
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance from model"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return self.plot_to_base64(plt)
    
    def plot_kenya_map(self):
        """Create an interactive map of accidents"""
        kenya_map = folium.Map(
            location=[-0.0236, 37.9062], 
            zoom_start=6,
            tiles='CartoDB positron'
        )
        
        # Add heatmap
        heat_data = [[row['latitude'], row['longitude']] 
                   for _, row in self.data.dropna(subset=['latitude', 'longitude']).iterrows()]
        HeatMap(heat_data).add_to(kenya_map)
        
        return kenya_map
    
    def plot_cluster_hotspots(self, kmeans, scaler):
        """Visualize accident clusters"""
        # Get cluster centers and transform back to original scale
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        hotspot_map = folium.Map(
            location=[-0.0236, 37.9062], 
            zoom_start=6,
            tiles='CartoDB positron'
        )
        
        # Add cluster centers
        for i, center in enumerate(centers):
            folium.Marker(
                location=[center[0], center[1]],
                icon=folium.Icon(color='red', icon='info-sign'),
                popup=f'Hotspot Cluster {i+1}'
            ).add_to(hotspot_map)
        
        return hotspot_map

class AccidentAnalyzer:
    def __init__(self, data):
        self.data = data
        self.visualizer = AccidentVisualizer(data)
    
    def temporal_analysis(self):
        """Analyze accident trends over time"""
        results = {}
        
        # Yearly trend
        yearly = self.data.groupby('year').size()
        results['yearly_trend'] = yearly
        results['yearly_trend_plot'] = self.visualizer.plot_temporal_trends({'yearly_trend': yearly})
        
        # Monthly pattern
        monthly = self.data.groupby('month').size()
        results['monthly_pattern'] = monthly
        
        # Day of week pattern
        dow = self.data.groupby('day_of_week').size()
        results['day_of_week_pattern'] = dow
        
        # Combine all for the temporal plot
        results['temporal_plot'] = self.visualizer.plot_temporal_trends({
            'yearly_trend': yearly,
            'monthly_pattern': monthly,
            'day_of_week_pattern': dow
        })
        
        return results
    
    def spatial_analysis(self):
        """Analyze accident patterns by location"""
        results = {}
        
        # By county
        by_county = self.data['county'].value_counts()
        results['by_county'] = by_county
        results['by_county_plot'] = self.visualizer.plot_spatial_distribution({'by_county': by_county})
        
        # By road type if available
        if 'road_type' in self.data.columns:
            by_road = self.data['road_type'].value_counts()
            results['by_road_type'] = by_road
            results['spatial_plot'] = self.visualizer.plot_spatial_distribution({
                'by_county': by_county,
                'by_road_type': by_road
            })
            
        return results
    
    def cluster_accident_hotspots(self, n_clusters=5):
        """Cluster accident locations to identify hotspots"""
        if 'latitude' not in self.data.columns or 'longitude' not in self.data.columns:
            raise ValueError("Latitude and longitude data required for clustering")
            
        coords = self.data[['latitude', 'longitude']].dropna()
        
        # Standardize coordinates
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(coords)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_coords)
        
        # Add cluster labels to data
        self.data.loc[coords.index, 'cluster'] = clusters
        
        return kmeans, scaler
    
    def predict_severity(self):
        """Predict accident severity with consistent return format"""
        try:
        # Ensure required columns exist with defaults
            required_cols = {
                'county': 'UNKNOWN',
                'road_type': 'UNKNOWN', 
                'time_of_day': 'Afternoon',
                'weather': 'UNKNOWN',
                'severity': 'No injury'
            }
        
            for col, default in required_cols.items():
                if col not in self.data.columns:
                    self.data[col] = default

        # Prepare features
            features = pd.get_dummies(self.data[['county', 'road_type', 'time_of_day', 'weather']])
            target = self.data['severity']

        # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=0.2,
                random_state=42
            )

        # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

        # Generate report
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

        # Return consistent format (single dictionary)
            return {
                'status': 'success',
                'model': model,
                'report': report,
                'features': list(features.columns),
                'sample_size': len(self.data)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
class KenyaAccidentAnalyzer:
    def __init__(self):
        self.preprocessor = AccidentDataPreprocessor()
        self.analyzer = None
        self.visualizer = None
    
    def load_and_preprocess_data(self, data):
        """Load and preprocess data for analysis"""
        if isinstance(data, str):
            # Assume it's a file path
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.xlsx'):
                df = pd.read_excel(data)
            else:
                raise ValueError("Unsupported file format")
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise TypeError("Data must be a DataFrame or file path")
        
        cleaned_data = self.preprocessor.clean_data(df)
        final_data = self.preprocessor.feature_engineering(cleaned_data)
        
        self.analyzer = AccidentAnalyzer(final_data)
        self.visualizer = AccidentVisualizer(final_data)
        
        return final_data
    
    def get_summary_stats(self):
        """Get summary statistics for the dashboard"""
        if self.analyzer is None:
            raise ValueError("No data loaded")
            
        return {
            'total_accidents': len(self.analyzer.data),
            'fatal_accidents': (self.analyzer.data['severity'] == 'Fatal').sum(),
            'top_county': self.analyzer.data['county'].value_counts().idxmax()
        }