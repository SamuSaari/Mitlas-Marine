from flask import Flask, request, render_template, send_from_directory
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for Matplotlib to prevent GUI warnings
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
import contextily as ctx
import os
from pyproj import CRS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'output')  # Updated to reflect the root directory structure
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create the directory if it doesn't exist

def detect_crs(x_values):
    if x_values.mean() > 2500000 and x_values.mean() < 2600000:
        return "EPSG:31469"  # GK25 (Zone 3)
    elif x_values.mean() > 2600000 and x_values.mean() < 2700000:
        return "EPSG:31470"  # GK26 (Zone 4)
    else:
        return "EPSG:4326"  # Default to WGS 84 if unrecognized

def plot_bathymetry_with_satellite_background(file_path, output_path, method='linear', grid_resolution=100, buffer=0.05):
    try:
        data = pd.read_csv(file_path, sep=',', header=None, usecols=[1, 2, 3], names=['X', 'Y', 'Z'])
        data['X'] = pd.to_numeric(data['X'], errors='coerce')
        data['Y'] = pd.to_numeric(data['Y'], errors='coerce')
        data['Z'] = pd.to_numeric(data['Z'], errors='coerce')
        data.dropna(inplace=True)

        detected_crs = detect_crs(data['X'].values)
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.X, data.Y), crs=detected_crs)
        gdf = gdf.to_crs(epsg=4326)

        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        z = data['Z'].values

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= x_range * buffer
        x_max += x_range * buffer
        y_min -= y_range * buffer
        y_max += y_range * buffer

        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method=method)

        gdf_wm = gdf.to_crs(epsg=3857)
        xi_wm, yi_wm = np.meshgrid(
            np.linspace(gdf_wm.geometry.x.min(), gdf_wm.geometry.x.max(), grid_resolution),
            np.linspace(gdf_wm.geometry.y.min(), gdf_wm.geometry.y.max(), grid_resolution)
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(gdf_wm.geometry.x.min(), gdf_wm.geometry.x.max())
        ax.set_ylim(gdf_wm.geometry.y.min(), gdf_wm.geometry.y.max())
        ctx.add_basemap(ax, crs=gdf_wm.crs.to_string(), source=ctx.providers.Esri.WorldImagery)
        depth_plot = ax.pcolormesh(xi_wm, yi_wm, zi, shading='auto', cmap='viridis', alpha=0.6)
        cbar = plt.colorbar(depth_plot, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Depth (meters)')
        ax.set_title('Bathymetry Depth Map with Satellite Background')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.savefig(output_path)
        plt.close()

        print(f"Image successfully saved to {output_path}")

    except Exception as e:
        print(f"Error generating plot: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the uploaded file in the 'static/output' directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Define the path for the output image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')

            try:
                plot_bathymetry_with_satellite_background(file_path, output_path)
            except Exception as e:
                print(f"Error generating plot: {e}")
                return "Error generating plot"

            return render_template('index.html', filename='output.png')
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
