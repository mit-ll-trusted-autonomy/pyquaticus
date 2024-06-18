import numpy as np
from PIL import Image
import folium
import io

# Function to convert Folium map to array of pixel values
def folium_map_to_array(folium_map):
    # Save map as PNG image
    img_data = folium_map._to_png()
    
    # Open PNG image using Pillow (PIL)
    image = Image.open(io.BytesIO(img_data))
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    return img_array

# Example usage
m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)  # Example map
image_array = folium_map_to_array(m)

# Now you have `image_array`, which contains the pixel values of the Folium map
print(image_array.shape)  # Shape of the array (height, width, channels)