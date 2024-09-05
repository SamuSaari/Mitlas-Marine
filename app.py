from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for Matplotlib to prevent GUI warnings
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
import contextily as ctx
import os
from threading import Timer
from shapely.affinity import translate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'output')  # Use a static directory for serving files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
ALLOWED_EXTENSIONS = {'txt'}  # Only allow .txt files

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_crs(data):
    """Detect whether the data is in latitude/longitude or a projected CRS like ETRS-GK."""
    x_min, x_max = data['X'].min(), data['X'].max()
    y_min, y_max = data['Y'].min(), data['Y'].max()
    
    # Check if the data is in latitude/longitude (EPSG:4326)
    if -180 <= x_min <= 180 and -180 <= x_max <= 180 and -90 <= y_min <= 90 and -90 <= y_max <= 90:
        print("Detected coordinate system: EPSG:4326 (WGS 84)")
        return "EPSG:4326"
    
    # Check if the data is in a Finland-specific ETRS-GK system
    if 3000000 <= x_min <= 7000000 and 5000000 <= y_min <= 8000000:
        print("Detected coordinate system: Finland ETRS-GK system (e.g., EPSG:3876 for GK26)")
        return "EPSG:3876"  # Default to GK26; adjust based on specific zone if needed
    
    # If data doesn't fit the criteria above, return None or a default
    print("Unable to detect the coordinate system. Defaulting to EPSG:4326.")
    return "EPSG:4326"  # Default to WGS 84

def plot_bathymetry(file_path, output_path, contours=False, depthmap=False, satellite=False, method='linear', grid_resolution=100, buffer=0.1):
    try:
        # Read the ASCII text file, skip the point number column
        data = pd.read_csv(file_path, sep=',', header=None, usecols=[1, 2, 3], names=['X', 'Y', 'Z'])
        data['X'] = pd.to_numeric(data['X'], errors='coerce')
        data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
        data['Z'] = pd.to_numeric(data['Z'], errors='coerce')
        data.dropna(inplace=True)

        if data.empty:
            raise ValueError("Invalid data: The uploaded file does not contain valid numeric data.")

        # Detect the CRS of the input data
        detected_crs = detect_crs(data)
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs=detected_crs)

        # Convert coordinates to EPSG:4326 (latitude and longitude) if not already in it
        if detected_crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Remove NaN or duplicate points after transformation
        gdf = gdf.dropna(subset=['geometry'])
        gdf = gdf.drop_duplicates(subset=['geometry'])

        # Apply small noise to avoid exact duplicates using shapely.affinity.translate
        gdf.geometry = gdf.geometry.apply(lambda geom: translate(
            geom, xoff=np.random.uniform(-1e-6, 1e-6), yoff=np.random.uniform(-1e-6, 1e-6)
        ))

        # Extract x, y, z coordinates
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        z = data['Z'].values

        # Determine grid extents with buffer for zooming out
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * buffer
        x_max += x_range * buffer
        y_min -= y_range * buffer
        y_max += y_range * buffer

        # Create grid for interpolation
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method=method)

        # Convert to Web Mercator for basemap
        gdf_wm = gdf.to_crs(epsg=3857)
        xi_wm, yi_wm = np.meshgrid(
            np.linspace(gdf_wm.geometry.x.min(), gdf_wm.geometry.x.max(), grid_resolution),
            np.linspace(gdf_wm.geometry.y.min(), gdf_wm.geometry.y.max(), grid_resolution)
        )

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 10))

        if satellite:
            ax.set_xlim(gdf_wm.geometry.x.min(), gdf_wm.geometry.x.max())
            ax.set_ylim(gdf_wm.geometry.y.min(), gdf_wm.geometry.y.max())
            ctx.add_basemap(ax, crs=gdf_wm.crs.to_string(), source=ctx.providers.Esri.WorldImagery)

        if depthmap:
            depth_plot = ax.pcolormesh(xi_wm, yi_wm, zi, shading='auto', cmap='viridis', alpha=0.6)
            cbar = plt.colorbar(depth_plot, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Depth (meters)')
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

        if contours:
            contours_plot = ax.contour(xi_wm, yi_wm, zi, levels=10, colors='black', linewidths=0.5)
            ax.clabel(contours_plot, fmt='%1.3f')  # Format contour labels with three decimals

        ax.set_title('Bathymetry Visualization')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Save the figure to the output path
        plt.savefig(output_path)
        plt.close()

        print(f"Image successfully saved to {output_path}")

    except Exception as e:
        print(f"Error generating plot: {e}")
        raise

def delete_file_after_delay(filepath, delay=300):
    """Delete a file after a delay (default 5 minutes)."""
    def delete_file():
        try:
            os.remove(filepath)
            print(f"Deleted file: {filepath}")
        except Exception as e:
            print(f"Error deleting file {filepath}: {e}")
    
    Timer(delay, delete_file).start()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            # Save the uploaded file in the specified static folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Get user options from the form
            contours = 'contours' in request.form
            depthmap = 'depthmap' in request.form
            satellite = 'satellite' in request.form

            # Define the path for the output image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')

            try:
                plot_bathymetry(file_path, output_path, contours=contours, depthmap=depthmap, satellite=satellite)
            except ValueError as e:
                return str(e)  # Return the error message to the user
            except Exception as e:
                print(f"Error generating plot: {e}")
                return "Error generating plot"

            # Schedule the uploaded file and output file for deletion after 5 minutes
            delete_file_after_delay(file_path)
            delete_file_after_delay(output_path)

            return render_template('index.html', filename='output.png')
        else:
            return "Invalid file type. Only .txt files are allowed."
    return render_template('index.html')

@app.route('/static/output/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
