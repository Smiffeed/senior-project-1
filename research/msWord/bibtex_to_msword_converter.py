import re
import html
import uuid
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

def parse_authors(author_string):
    """Parse author string and return MS Word format"""
    authors = []
    # Split by 'and' to get individual authors
    author_list = re.split(r'\s+and\s+', author_string)
    
    for author in author_list:
        author = author.strip()
        # Try to split "Last, First" format
        if ',' in author:
            parts = author.split(',', 1)
            last = parts[0].strip()
            first = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Try to split "First Last" format
            parts = author.split()
            if len(parts) >= 2:
                first = ' '.join(parts[:-1])
                last = parts[-1]
            else:
                first = ""
                last = author
        
        authors.append({'first': first, 'last': last})
    
    return authors

def bibtex_to_msword_xml(bibtex_path, xml_path):
    """Convert BibTeX file to MS Word XML format"""
    with open(bibtex_path, 'r', encoding='utf-8') as f:
        bibtex_content = f.read()
    
    # Find all entries
    entries = []
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
    
    # Start building MS Word XML
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<b:Sources SelectedStyle="\\APASixthEditionOfficeOnline.xsl" StyleName="APA" Version="6" xmlns:b="http://schemas.openxmlformats.org/officeDocument/2006/bibliography" xmlns="http://schemas.openxmlformats.org/officeDocument/2006/bibliography">
'''
    
    # Parse entries and convert to MS Word XML
    for entry_text in entries:
        parsed = parse_bibtex_entry(entry_text)
        if not parsed:
            continue
        
        entry_type, entry_key, fields = parsed
        
        # Generate a unique GUID for this source
        guid = str(uuid.uuid4()).upper()
        
        xml_content += f'  <b:Source>\n'
        xml_content += f'    <b:Tag>{entry_key}</b:Tag>\n'
        
        # Map BibTeX entry types to MS Word source types
        source_type_map = {
            'article': 'ArticleInAPeriodical',
            'book': 'Book',
            'inproceedings': 'ConferenceProceedings',
            'incollection': 'BookSection',
            'phdthesis': 'Report',
            'mastersthesis': 'Report',
            'techreport': 'Report',
            'misc': 'Misc'
        }
        
        source_type = source_type_map.get(entry_type.lower(), 'Misc')
        xml_content += f'    <b:SourceType>{source_type}</b:SourceType>\n'
        xml_content += f'    <b:GUID>{{{guid}}}</b:GUID>\n'
        
        # Handle authors
        if 'author' in fields:
            authors = parse_authors(fields['author'])
            xml_content += '    <b:Author>\n      <b:Author>\n        <b:NameList>\n'
            for author in authors:
                xml_content += '          <b:Person>\n'
                xml_content += f'            <b:Last>{html.escape(author["last"])}</b:Last>\n'
                xml_content += f'            <b:First>{html.escape(author["first"])}</b:First>\n'
                xml_content += '          </b:Person>\n'
            xml_content += '        </b:NameList>\n      </b:Author>\n    </b:Author>\n'
        
        # Handle other fields
        field_map = {
            'title': 'Title',
            'journal': 'PeriodicalTitle',
            'booktitle': 'ConferenceName',
            'year': 'Year',
            'month': 'Month',
            'volume': 'Volume',
            'number': 'Issue',
            'pages': 'Pages',
            'publisher': 'Publisher',
            'address': 'City',
            'doi': 'DOI',
            'url': 'URL',
            'isbn': 'StandardNumber'
        }
        
        for bib_field, ms_field in field_map.items():
            if bib_field in fields:
                value = html.escape(fields[bib_field])
                xml_content += f'    <b:{ms_field}>{value}</b:{ms_field}>\n'
        
        xml_content += '  </b:Source>\n'
    
    xml_content += '</b:Sources>'
    
    # Write to file
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

def convert_all_bibtex_to_msword(directory):
    """Convert all .bib files in a directory to MS Word XML"""
    directory = Path(directory)
    bib_files = list(directory.glob('*.bib'))
    
    if not bib_files:
        print(f"No .bib files found in {directory}")
        return
    
    for bib_file in bib_files:
        xml_file = bib_file.with_name(bib_file.stem + '_msword.xml')
        try:
            bibtex_to_msword_xml(str(bib_file), str(xml_file))
            print(f"✅ Converted {bib_file.name} to {xml_file.name} (MS Word format)")
        except Exception as e:
            print(f"❌ Error converting {bib_file.name}: {e}")

if __name__ == "__main__":
    # Convert all BibTeX files to MS Word XML format
    convert_all_bibtex_to_msword("research/msWord")