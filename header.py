import re
import numpy as np

def parse_h_matrix(file_path):
    """
    Parses a C++ header file matrix into a clean Python list.
    Handles { }, nested arrays, and 'f' suffixes.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Find the data inside the outermost curly braces
    # This regex looks for the content following the '=' sign
    match = re.search(r'=\s*\{([\s\S]*)\};', content)
    if not match:
        # Fallback: just look for anything between braces if '=' isn't found
        match = re.search(r'\{([\s\S]*)\}', content)
    
    if not match:
        raise ValueError(f"Could not find matrix data in {file_path}")

    raw_data = match.group(1)

    # 2. Cleanup C++ specific syntax
    # Remove 'f' suffixes (e.g., 0.123f -> 0.123) 
    clean_data = re.sub(r'([0-9.]+)[fF]', r'\1', raw_data)
    # Convert curly braces to square brackets for Python eval
    clean_data = clean_data.replace('{', '[').replace('}', ']')
    # Remove C++ comments if any exist
    clean_data = re.sub(r'//.*', '', clean_data)

    try:
        # Use eval to transform the string into a nested Python list
        return eval(clean_data)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def create_config_file(mel_h, dct_h, output_file="features_config.py"):
    mel_matrix = parse_h_matrix(mel_h)
    dct_matrix = parse_h_matrix(dct_h)

    with open(output_file, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write(f"# Processed from {mel_h}\n")
        f.write(f"MEL_FILTER_BANK = np.array({mel_matrix}, dtype=np.float32)\n\n")
        f.write(f"# Processed from {dct_h}\n")
        f.write(f"DCT_MATRIX = np.array({dct_matrix}, dtype=np.float32)\n")
    
    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    # Ensure these match your actual filenames
    create_config_file("mel_filter_bank.h", "dct_weights.h")