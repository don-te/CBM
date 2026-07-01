import os
import math
import folium
import pandas as pd
from branca.element import Template, MacroElement

# --- CONFIGURATION ---
CSV_PATH = "./data/cbm_batch_results/cbm_detections_log.csv"
OUTPUT_MAP = "./data/cbm_batch_results/cbm_interactive_map.html"

# Sentinel-2 constants
PATCH_SIZE_PX = 128     # The original patch size your CSV boxes were scaled back to
PIXEL_RESOLUTION_M = 10 # 10 meters per pixel

def pixel_to_gps(center_lat, center_lon, x_min, y_min, x_max, y_max):
    """Converts image pixel coordinates into exact real-world GPS coordinates."""
    px = (x_min + x_max) / 2.0
    py = (y_min + y_max) / 2.0
    
    dx_px = px - (PATCH_SIZE_PX / 2.0)
    dy_px = py - (PATCH_SIZE_PX / 2.0)
    
    dlat = -dy_px * PIXEL_RESOLUTION_M / 111320.0 
    dlon = dx_px * PIXEL_RESOLUTION_M / (111320.0 * math.cos(math.radians(center_lat)))
    
    return center_lat + dlat, center_lon + dlon

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CRITICAL ERROR: Could not find {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    if len(df) == 0:
        print("CSV is empty. No detections to map.")
        return

    print(f"Loading {len(df)} detections onto the map...")

    # Get a starting location from the first row
    first_file = df.iloc[0]['image'].replace('.png', '').replace('.tif', '').replace('.jpg', '')
    try:
        start_lat, start_lon = map(float, first_file.split('_'))
    except ValueError:
        start_lat, start_lon = 28.05, 77.28 # Fallback to Palwal center
        
    m = folium.Map(location=[start_lat, start_lon], zoom_start=11)

    # Add Basemaps
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False, control=True
    ).add_to(m)
    folium.TileLayer('OpenStreetMap', name='OSM (Roads/Labels)', overlay=False, control=True).add_to(m)

    # Feature Groups so you can toggle concepts on and off
    fg_dict = {
        "industrial chimney": (folium.FeatureGroup(name="Micro: Industrial Chimney"), '#e74c3c'), # Red
        "tall smokestack": (folium.FeatureGroup(name="Micro: Tall Smokestack"), '#e67e22'),      # Orange
        "heavy shadow cast": (folium.FeatureGroup(name="Macro: Heavy Shadow Cast"), '#3498db'),  # Blue
        "large industrial roof": (folium.FeatureGroup(name="Macro: Industrial Roof"), '#9b59b6'),# Purple
        "cleared barren land": (folium.FeatureGroup(name="Macro: Barren Land"), '#2ecc71')       # Green
    }

    # Plot each detection
    success_count = 0
    for _, row in df.iterrows():
        filename = row['image']
        
        # THE FIX: Safely strip the extension without breaking the decimal coordinates
        stem = os.path.splitext(filename)[0] 
        
        try:
            # Clean up potential prefixes if they exist
            clean_stem = stem.replace('Tile_', '').replace('tile_', '')
            patch_lat, patch_lon = map(float, clean_stem.split('_'))
        except ValueError:
            continue # Skip files that still don't match lat_lon format
            
        exact_lat, exact_lon = pixel_to_gps(
            patch_lat, patch_lon, 
            row['xmin'], row['ymin'], row['xmax'], row['ymax']
        )

        concept = row['concept']
        conf = row['confidence']
        
        if concept in fg_dict:
            fg, color = fg_dict[concept]
            
            popup_text = f"<b>OWLv2 CBM Detection</b><br>Concept: <b>{concept}</b><br>Confidence: {conf}<br>File: {filename}"
            
            folium.CircleMarker(
                location=[exact_lat, exact_lon],
                radius=5,
                color="white",
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(fg)
            
            success_count += 1

    # Add groups to map
    for concept, (fg, color) in fg_dict.items():
        fg.add_to(m)
    
    # Add custom legend
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: rgba(255, 255, 255, 0.95); border: 1px solid #444; border-radius: 6px; padding: 12px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        <b>CBM Concept Detections</b><br>
        <i>Toggle layers top-right</i><br><hr style="margin: 4px 0;">
        <b>Micro Concepts (High Hallucination Risk)</b><br>
        <span style="color:#e74c3c;">&#9679;</span> Industrial Chimney<br>
        <span style="color:#e67e22;">&#9679;</span> Tall Smokestack<br>
        <hr style="margin: 4px 0;">
        <b>Macro Concepts (Primitive Geometries)</b><br>
        <span style="color:#3498db;">&#9679;</span> Heavy Shadow Cast<br>
        <span style="color:#9b59b6;">&#9679;</span> Large Industrial Roof<br>
        <span style="color:#2ecc71;">&#9679;</span> Cleared Barren Land<br>
    </div>
    '''
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)

    folium.LayerControl().add_to(m)
    m.save(OUTPUT_MAP)
    
    print(f"SUCCESS: Interactive map generated at {OUTPUT_MAP}")
    print(f"Plotted {success_count} concepts across Palwal.")

if __name__ == "__main__":
    main()