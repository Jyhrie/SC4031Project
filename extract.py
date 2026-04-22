with open('test.txt', 'r', encoding='utf-8') as f_in, \
     open('out.txt', 'w', encoding='utf-8') as f_out:
    
    # 1. Read and strip whitespace
    # 2. Remove 's from each line
    # 3. Ignore empty lines
    lines = [line.strip().replace("'", "") for line in f_in if line.strip()]
    
    # Wrap in quotes and join with commas
    formatted_content = ", ".join(f'"{line}"' for line in lines)
    
    f_out.write(formatted_content)

print("Successfully generated out.txt (UTF-8) with 's removed.")