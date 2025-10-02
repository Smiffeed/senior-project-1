import re
import html
from pathlib import Path

def parse_bibtex_entry(entry_text):
    """Parse a single BibTeX entry and return its components"""
    # Extract entry type and key
    match = re.match(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', entry_text)
    if not match:
        return None
    
    entry_type = match.group(1)
    entry_key = match.group(2)
    
    # Find the content after the key
    content_start = match.end()
    content = entry_text[content_start:].rstrip(' }')
    
    # Parse fields
    fields = {}
    i = 0
    while i < len(content):
        # Skip whitespace and commas
        while i < len(content) and content[i] in ' \n\t,':
            i += 1
        
        if i >= len(content):
            break
        
        # Find field name
        field_name_match = re.match(r'(\w+)\s*=\s*', content[i:])
        if not field_name_match:
            i += 1
            continue
        
        field_name = field_name_match.group(1)
        i += field_name_match.end()
        
        # Find field value
        if i < len(content) and content[i] == '{':
            # Handle braced values
            brace_count = 0
            start = i + 1
            i += 1
            
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    if brace_count == 0:
                        break
                    brace_count -= 1
                i += 1
            
            if i < len(content):
                field_value = content[start:i]
                fields[field_name] = field_value.strip()
                i += 1
        else:
            # Handle unbraced values (numbers, etc.)
            start = i
            while i < len(content) and content[i] not in ',}':
                i += 1
            field_value = content[start:i].strip()
            fields[field_name] = field_value
    
    return entry_type, entry_key, fields

def bibtex_to_xml(bibtex_path, xml_path):
    """Convert BibTeX file to XML format"""
    with open(bibtex_path, 'r', encoding='utf-8') as f:
        bibtex_content = f.read()
    
    # Find all entries
    entries = []
    entry_pattern = r'@\w+\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    # Use a more sophisticated approach to find entries
    i = 0
    while i < len(bibtex_content):
        # Find start of entry
        if bibtex_content[i] == '@':
            start = i
            # Find the opening brace
            while i < len(bibtex_content) and bibtex_content[i] != '{':
                i += 1
            
            if i >= len(bibtex_content):
                break
            
            # Count braces to find the end
            brace_count = 1
            i += 1
            
            while i < len(bibtex_content) and brace_count > 0:
                if bibtex_content[i] == '{':
                    brace_count += 1
                elif bibtex_content[i] == '}':
                    brace_count -= 1
                i += 1
            
            if brace_count == 0:
                entry_text = bibtex_content[start:i]
                entries.append(entry_text)
        else:
            i += 1
    
    # Parse entries and convert to XML
    xml_entries = []
    for entry_text in entries:
        parsed = parse_bibtex_entry(entry_text)
        if parsed:
            entry_type, entry_key, fields = parsed
            xml_entry = f'  <entry type="{entry_type}" key="{entry_key}">\n'
            
            for field_name, field_value in fields.items():
                # Clean up the field value
                field_value = html.unescape(field_value)
                field_value = html.escape(field_value)
                xml_entry += f'    <{field_name}>{field_value}</{field_name}>\n'
            
            xml_entry += '  </entry>\n'
            xml_entries.append(xml_entry)
    
    # Create XML content
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<bibtex>\n' + ''.join(xml_entries) + '</bibtex>\n'
    
    # Write to file
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

def convert_all_bibtex_files(directory):
    """Convert all .bib files in a directory to XML"""
    directory = Path(directory)
    bib_files = list(directory.glob('*.bib'))
    
    if not bib_files:
        print(f"No .bib files found in {directory}")
        return
    
    for bib_file in bib_files:
        xml_file = bib_file.with_suffix('.xml')
        try:
            bibtex_to_xml(str(bib_file), str(xml_file))
            print(f"✅ Converted {bib_file.name} to {xml_file.name}")
        except Exception as e:
            print(f"❌ Error converting {bib_file.name}: {e}")

if __name__ == "__main__":
    # Convert all BibTeX files in the research/msWord directory
    convert_all_bibtex_files("research/msWord")