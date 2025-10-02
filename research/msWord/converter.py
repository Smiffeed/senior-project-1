import re
import html

def bibtex_to_xml(bibtex_path, xml_path):
    with open(bibtex_path, 'r', encoding='utf-8') as f:
        bibtex = f.read()

    # Clean up the bibtex content
    bibtex = bibtex.strip()
    
    # Find all entries using a more robust pattern
    entry_pattern = r'@(\w+)\s*\{\s*([^,\s]+)\s*,\s*(.*?)\s*\}'
    entries = re.findall(entry_pattern, bibtex, re.DOTALL)
    
    xml_entries = []

    for entry_type, entry_key, fields in entries:
        xml_entry = f'  <entry type="{entry_type}" key="{entry_key}">\n'
        
        # Parse fields more carefully
        field_pattern = r'(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*,?'
        field_matches = re.findall(field_pattern, fields, re.DOTALL)
        
        for field_name, field_value in field_matches:
            # Clean up the field value
            field_value = field_value.strip()
            # Handle HTML entities
            field_value = html.unescape(field_value)
            # Escape XML special characters
            field_value = html.escape(field_value)
            
            xml_entry += f'    <{field_name}>{field_value}</{field_name}>\n'
        
        xml_entry += '  </entry>\n'
        xml_entries.append(xml_entry)

    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<bibtex>\n' + ''.join(xml_entries) + '</bibtex>\n'

    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

if __name__ == "__main__":
    # Convert the citation file
    bibtex_to_xml("research/msWord/citation-363796345.bib", "research/msWord/citation-363796345.xml")
    print("Converted citation-363796345.bib to XML")
    
    # Convert the hate speech detection file
    bibtex_to_xml("research/msWord/Automated-Hate-Speech-Detection-and-the-Problem-of-Offensive.bib", 
                  "research/msWord/Automated-Hate-Speech-Detection-and-the-Problem-of-Offensive.xml")
    print("Converted Automated-Hate-Speech-Detection-and-the-Problem-of-Offensive.bib to XML")