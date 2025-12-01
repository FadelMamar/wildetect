# Class Mapping Configuration

> **Location**: `config/class_mapping.json`

**Purpose**: JSON file that maps class IDs (numeric) to class names (strings) for wildlife species and other detected objects. This mapping is used throughout the detection pipeline to convert between numeric class IDs and human-readable names.

## Configuration Structure

### Complete Parameter Reference

```json
{
  "0": "Tsessebe",
  "1": "Waterbuck",
  "2": "buffalo",
  "3": "bushbuck",
  "4": "colour impala",
  "5": "duiker",
  "6": "giraffe",
  "7": "impala",
  "8": "kudu",
  "9": "label",
  "10": "lechwe",
  "11": "nyala",
  "12": "nyala(m)",
  "13": "other",
  "14": "other animal",
  "15": "reedbuck",
  "16": "roan",
  "17": "rocks",
  "18": "sable",
  "19": "termite mound",
  "20": "vegetation",
  "21": "warthog",
  "22": "wildebeest",
  "23": "zebra",
  "24": "wildlife"
}
```

### Format Description

The class mapping is a JSON object where:
- **Keys**: String representations of class IDs (e.g., `"0"`, `"1"`, `"2"`)
- **Values**: Class names as strings (e.g., `"elephant"`, `"giraffe"`, `"zebra"`)

### Parameter Descriptions

Each entry in the mapping follows the format:
```json
"<class_id>": "<class_name>"
```

- **`<class_id>`**: Numeric class ID used by the model (as string key)
- **`<class_name>`**: Human-readable name for the class

**Common Classes in Wildlife Detection**:
- Wildlife species: `"elephant"`, `"giraffe"`, `"zebra"`, `"buffalo"`, `"wildebeest"`, `"impala"`, etc.
- Background objects: `"vegetation"`, `"rocks"`, `"termite mound"`, `"other"`
- Generic categories: `"wildlife"`, `"other animal"`, `"label"`

---

## Example Configurations

### Minimal Class Mapping

```json
{
  "0": "elephant",
  "1": "giraffe",
  "2": "zebra",
  "3": "buffalo",
  "4": "wildebeest"
}
```

### Extended Wildlife Mapping

```json
{
  "0": "elephant",
  "1": "giraffe",
  "2": "zebra",
  "3": "buffalo",
  "4": "wildebeest",
  "5": "impala",
  "6": "kudu",
  "7": "waterbuck",
  "8": "warthog",
  "9": "vegetation",
  "10": "rocks",
  "11": "other"
}
```

### Mapping with Background Classes

```json
{
  "0": "elephant",
  "1": "giraffe",
  "2": "zebra",
  "3": "buffalo",
  "4": "vegetation",
  "5": "rocks",
  "6": "termite_mound",
  "7": "other"
}
```

---

## Best Practices

1. **Consistent Naming**: Use consistent naming conventions (e.g., lowercase, snake_case, or camelCase)
2. **Complete Mapping**: Ensure all class IDs used by your model are included in the mapping
3. **No Gaps**: Class IDs should be sequential starting from 0 (0, 1, 2, 3, ...)
4. **Descriptive Names**: Use clear, descriptive names that match your dataset annotations
5. **Case Sensitivity**: Be aware that class names are case-sensitive
6. **Special Characters**: Avoid special characters that might cause issues in file paths or URLs
7. **Version Control**: Keep the mapping file in version control and update it when model classes change

---

## Usage in Code

The class mapping is typically loaded and used as follows:

```python
import json

# Load class mapping
with open('config/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

# Convert class ID to name
class_id = 0
class_name = class_mapping[str(class_id)]  # "Tsessebe"

# Convert name to ID (reverse lookup)
class_name = "giraffe"
class_id = [k for k, v in class_mapping.items() if v == class_name][0]  # "6"
```

---

## Troubleshooting

### Class ID Not Found

**Issue**: Error when looking up class ID in mapping

**Solutions**:
1. Verify class ID exists in the mapping
2. Ensure class IDs are strings (JSON keys are always strings)
3. Check for typos in class ID
4. Add missing class IDs to the mapping

### Class Name Mismatch

**Issue**: Class names don't match between mapping and annotations

**Solutions**:
1. Verify class names match exactly (case-sensitive)
2. Check for whitespace or special characters
3. Ensure mapping matches the training dataset class names
4. Update mapping to match your dataset

### Missing Classes

**Issue**: Model predicts classes not in the mapping

**Solutions**:
1. Add missing class IDs and names to the mapping
2. Verify model was trained with the same class set
3. Check if class IDs are zero-indexed (starting from 0)
4. Ensure mapping covers all model output classes

### JSON Parse Error

**Issue**: Cannot parse class_mapping.json

**Solutions**:
1. Validate JSON syntax using a JSON validator
2. Check for trailing commas
3. Ensure all keys and values are properly quoted
4. Verify file encoding is UTF-8

---

## Related Documentation

- [Configuration Overview](../index.md)
- [Detection Config](detection.md)
- [Census Config](census.md)
- [Model Training](../../tutorials/model-training.md)

