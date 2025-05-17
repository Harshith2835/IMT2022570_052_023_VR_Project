# Import required libraries for JSON processing, CSV handling, and time measurement
import json
import csv
import time

# Define input and output file paths
INPUT_LISTINGS_FILE = "listings_0.json"  # Source file containing listing data
INPUT_IMAGES_FILE = "images.csv"        # Source file containing image metadata
OUTPUT_FILE = "cleaned_vqa_metadata_with_images.json"  # Output file for cleaned data

# Dictionary defining fields to extract from listings, with a flag indicating if the field is required
SIMPLE_FIELDS = {
    "brand": False,         # Brand name of the item
    "item_name": False,     # Name of the item
    "model_name": False,    # Model name of the item
    "model_number": False,  # Model number of the item
    "style": False,         # Style of the item
    "color": False,         # Color of the item
    "color_code": True,     # Color code (required field)
    "item_id": True,        # Unique item identifier (required field)
    "model_year": False,    # Model year of the item
    "product_type": False,  # Type of product
    "bullet_point": False,  # Bullet points describing the item
    "item_keywords": False, # Keywords associated with the item
}

# Function to filter and extract values based on language tags
def pick_values(field_list, keep_tags=("en_",)):
    """
    Extracts values from a list of field entries, keeping only those with no language tag
    or with tags starting with specified prefixes (e.g., 'en_').
    
    Args:
        field_list: List of dictionaries containing field values and language tags
        keep_tags: Tuple of language tag prefixes to include (default: English tags)
    
    Returns:
        List of extracted values
    """
    out = []
    for entry in field_list:
        tag = entry.get("language_tag")
        if tag is None:  # Include value if no language_tag
            out.append(entry["value"])
        elif tag.startswith("en_"):  # Include value if tag starts with "en_"
            out.append(entry["value"])
    return out

# Function to flatten and process field values from raw JSON data
def flatten(item, field):
    """
    Processes a field from the raw JSON item, extracting and formatting values as needed.
    
    Args:
        item: Dictionary containing raw JSON data for an item
        field: Field name to process
    
    Returns:
        Processed field value, or None if the field is missing or empty
    """
    if field not in item:
        return None
    val = item[field]
    if field in ("color_code", "item_id", "main_image_id"):
        return val  # Return raw value for these fields
    arr = pick_values(val)
    if not arr:
        return None
    if field == "model_year":
        # Convert model_year to integer
        return arr[0] if isinstance(arr[0], int) else int(arr[0])
    if field in ("brand", "item_name", "model_name", "model_number", "style", "color", "product_type"):
        return arr[0]  # Return first value for single-value fields
    return arr  # Return array for multi-value fields like bullet points

# Record start time for performance measurement
start_time = time.time()

# Step 1: Read image metadata into a hashmap for quick lookup
image_map = {}
with open(INPUT_IMAGES_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Store image metadata (height, width, path) keyed by image_id
        image_map[row["image_id"]] = {
            "height": int(row["height"]),
            "width": int(row["width"]),
            "path": row["path"]
        }

# Step 2: Process listings and attach image metadata
cleaned = {}  # Dictionary to store cleaned data
line_count = 0  # Counter for processed lines

with open(INPUT_LISTINGS_FILE, "r", encoding="utf-8") as fin:
    for line in fin:
        # Parse each line as JSON
        raw = json.loads(line)
        img_id = raw.get("main_image_id")
        if not img_id:
            line_count += 1
            continue  # Skip entries without a main image ID

        entry = {}  # Dictionary to store processed fields for this item
        # Extract and process each field defined in SIMPLE_FIELDS
        for fld, _ in SIMPLE_FIELDS.items():
            val = flatten(raw, fld)
            if val is not None:
                # Pluralize key for bullet_point and item_keywords
                key = fld + ("s" if fld in ("bullet_point", "item_keywords") else "")
                entry[key] = val

        entry["item_id"] = raw.get("item_id")  # Always include item_id

        # Extract node name if available
        nodes = raw.get("node", [])
        if nodes and isinstance(nodes, list):
            entry["node_name"] = nodes[0].get("node_name")

        # Use bullet points as image description source
        entry["describe_image_source"] = entry.get("bullet_points", [])

        # Add image metadata for main and other images
        image_metadata = {}
        if img_id in image_map:
            image_metadata[img_id] = image_map[img_id]

        for other_id in raw.get("other_image_id", []):
            if other_id in image_map:
                image_metadata[other_id] = image_map[other_id]

        if image_metadata:
            entry["image_metadata"] = image_metadata

        cleaned[img_id] = entry  # Store entry keyed by main image ID
        line_count += 1

# Step 3: Write cleaned data to output JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    json.dump(cleaned, fout, indent=2)  # Write with indentation for readability

# Calculate and display execution time
end_time = time.time()
execution_time = end_time - start_time

print(f"Cleaned metadata with image info written to {OUTPUT_FILE}")
print(f"Execution time: {execution_time:.2f} seconds")