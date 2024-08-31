import base64

# Path to your image
image_path = "teeth_image.png"
# Path to the output text file
output_path = "teeth_image_base64.txt"

# Read the image and encode it to base64
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Save the base64 string to a text file
with open(output_path, "w") as text_file:
    text_file.write(encoded_string)

print(f"Base64 string saved to {output_path}")