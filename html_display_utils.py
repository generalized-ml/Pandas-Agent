from IPython.display import HTML, display
import re

def format_agent_output_to_html(result) -> str:
    """
    Convert agent output to formatted HTML.
    
    Args:
        result: The result object from the agent invocation
    
    Returns:
        HTML string with formatted output
    """
    # Extract content blocks
    content_blocks = result["messages"][-1].content_blocks
    
    html_parts = []
    html_parts.append("""
    <style>
        .agent-output {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .agent-header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .agent-header h2 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 24px;
        }
        .agent-header p {
            margin: 0;
            color: #666;
            font-size: 14px;
        }
        .content-block {
            background: white;
            padding: 25px;
            margin: 15px 0;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .content-block h3 {
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            font-size: 20px;
        }
        .content-block h4 {
            color: #555;
            margin-top: 15px;
            font-size: 16px;
        }
        .metric-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .metric-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        .metric-table td {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        .metric-table tr:hover {
            background: #f8f9fa;
        }
        .metric-table tr:last-child td {
            border-bottom: none;
        }
        .highlight {
            background: #fff3cd;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
            color: #856404;
        }
        .success {
            color: #28a745;
            font-weight: bold;
        }
        .warning {
            color: #ffc107;
            font-weight: bold;
        }
        .error {
            color: #dc3545;
            font-weight: bold;
        }
        .emoji {
            font-size: 20px;
            margin-right: 8px;
        }
        .insight-box {
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
        }
        .insight-box h4 {
            margin-top: 0;
            color: #007bff;
        }
        .insight-box ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .separator {
            border: 0;
            height: 2px;
            background: linear-gradient(to right, transparent, #667eea, transparent);
            margin: 25px 0;
        }
    </style>
    """)
    
    html_parts.append('<div class="agent-output">')
    html_parts.append('<div class="agent-header">')
    html_parts.append('<h2>🤖 AI Sales Agent Response</h2>')
    html_parts.append('<p>Generated analysis and insights</p>')
    html_parts.append('</div>')
    
    for i, block in enumerate(content_blocks, 1):
        html_parts.append('<div class="content-block">')
        
        # Handle both dictionary and object formats
        if isinstance(block, dict):
            text = block.get('text', '')
            block_type = block.get('type', 'text')
        elif hasattr(block, 'text'):
            text = block.text
            block_type = getattr(block, 'type', 'text')
        else:
            continue
        
        # Check if it's a tool call
        if isinstance(block, dict) and 'name' in block:
            html_parts.append(f'<div class="tool-call">')
            html_parts.append(f'<span class="tool-name">🔧 Tool: {block["name"]}</span>')
            html_parts.append(f'</div>')
        elif hasattr(block, 'name'):
            html_parts.append(f'<div class="tool-call">')
            html_parts.append(f'<span class="tool-name">🔧 Tool: {block.name}</span>')
            html_parts.append(f'</div>')
        
        # Format the text content with markdown support
        html_content = format_markdown_to_html(text)
        html_parts.append(html_content)
        
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    return ''.join(html_parts)


def format_markdown_to_html(text: str) -> str:
    """
    Convert markdown text to formatted HTML with tables and formatting support.
    
    Args:
        text: Markdown text content
    
    Returns:
        Formatted HTML string
    """
    lines = text.split('\n')
    html_lines = []
    in_table = False
    table_headers = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            html_lines.append('<br>')
            i += 1
            continue
        
        # Handle markdown headers (### Header)
        if stripped.startswith('###'):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            header_text = stripped.replace('###', '').strip()
            html_lines.append(f'<h3>{header_text}</h3>')
            i += 1
            continue
        
        # Handle horizontal rules (---)
        if stripped.startswith('---'):
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
            html_lines.append('<hr class="separator">')
            i += 1
            continue
        
        # Detect markdown tables (| Header | Header |)
        if '|' in stripped:
            if not in_table:
                # Start of table
                cols = [col.strip() for col in stripped.split('|') if col.strip()]
                table_headers = cols
                html_lines.append('<table class="metric-table">')
                html_lines.append('<thead><tr>')
                for col in cols:
                    html_lines.append(f'<th>{highlight_markdown(col)}</th>')
                html_lines.append('</tr></thead>')
                
                # Check next line for separator (|---|---|)
                if i + 1 < len(lines) and '|' in lines[i + 1] and '-' in lines[i + 1]:
                    i += 1  # Skip separator line
                
                html_lines.append('<tbody>')
                in_table = True
                i += 1
                continue
            else:
                # Table row
                cols = [col.strip() for col in stripped.split('|') if col.strip()]
                html_lines.append('<tr>')
                for col in cols:
                    html_lines.append(f'<td>{highlight_markdown(col)}</td>')
                html_lines.append('</tr>')
                i += 1
                continue
        
        # If we were in a table but no more pipes, close it
        if in_table:
            html_lines.append('</tbody></table>')
            in_table = False
        
        # Handle bullet points
        if stripped.startswith('-') or stripped.startswith('*'):
            content = stripped[1:].strip()
            html_lines.append(f'<div class="insight-box"><p>{highlight_markdown(content)}</p></div>')
            i += 1
            continue
        
        # Regular paragraph
        html_lines.append(f'<p>{highlight_markdown(stripped)}</p>')
        i += 1
    
    if in_table:
        html_lines.append('</tbody></table>')
    
    return ''.join(html_lines)


def format_text_to_html(text: str) -> str:
    """
    Convert plain text to formatted HTML with smart detection of structure.
    
    Args:
        text: Plain text content
    
    Returns:
        Formatted HTML string
    """
    lines = text.split('\n')
    html_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<br>')
            continue
        
        # Headers (lines with all caps or ending with colon)
        if stripped.isupper() or (stripped.endswith(':') and len(stripped) < 50):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3>{stripped}</h3>')
        
        # List items (starting with -, •, numbers)
        elif re.match(r'^[-•*]\s', stripped) or re.match(r'^\d+\.\s', stripped):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            content = re.sub(r'^[-•*]\s', '', stripped)
            content = re.sub(r'^\d+\.\s', '', content)
            html_lines.append(f'<li>{highlight_values(content)}</li>')
        
        # Metric rows (containing colon)
        elif ':' in stripped:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            parts = stripped.split(':', 1)
            if len(parts) == 2:
                label, value = parts
                html_lines.append(f'<div class="metric-row">')
                html_lines.append(f'<span class="metric-label">{label.strip()}:</span>')
                html_lines.append(f'<span class="metric-value">{highlight_values(value.strip())}</span>')
                html_lines.append(f'</div>')
            else:
                html_lines.append(f'<p>{highlight_values(stripped)}</p>')
        
        # Section separators
        elif stripped.startswith('═') or stripped.startswith('─'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<hr style="border: 1px solid #ddd; margin: 15px 0;">')
        
        # Regular paragraphs
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<p>{highlight_values(stripped)}</p>')
    
    if in_list:
        html_lines.append('</ul>')
    
    return ''.join(html_lines)


def highlight_markdown(text: str) -> str:
    """
    Convert markdown formatting to HTML and highlight special values.
    
    Args:
        text: Text with markdown formatting
    
    Returns:
        HTML formatted text
    """
    # Handle bold markdown (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Highlight currency (e.g., $1,234.56)
    text = re.sub(r'(\$[\d,]+\.?\d*)', r'<span class="highlight">\1</span>', text)
    
    # Highlight percentages
    text = re.sub(r'([\d.]+%)', r'<span class="highlight">\1</span>', text)
    
    # Highlight positive growth
    text = re.sub(r'(\+[\d.]+%)', r'<span class="success">\1</span>', text)
    
    # Highlight negative values (but not in table separators)
    if not re.match(r'^[\s\-|]+$', text):
        text = re.sub(r'(-[\d.]+%)', r'<span class="error">\1</span>', text)
    
    return text


def highlight_values(text: str) -> str:
    """
    Highlight numerical values, percentages, and currency in text.
    
    Args:
        text: Text to highlight
    
    Returns:
        Text with HTML spans around important values
    """
    # Highlight currency (e.g., $1,234.56)
    text = re.sub(r'(\$[\d,]+\.?\d*)', r'<span class="highlight">\1</span>', text)
    
    # Highlight percentages
    text = re.sub(r'([\d.]+%)', r'<span class="highlight">\1</span>', text)
    
    # Highlight positive growth
    text = re.sub(r'(\+[\d.]+%)', r'<span class="success">\1</span>', text)
    
    # Highlight negative values
    text = re.sub(r'(-[\d.]+%)', r'<span class="error">\1</span>', text)
    
    return text


def display_agent_output_html(result):
    """
    Display agent output in formatted HTML in Jupyter notebook.
    
    Args:
        result: The result object from the agent invocation
    """
    html_output = format_agent_output_to_html(result)
    display(HTML(html_output))


def save_agent_output_html(result, filename='agent_output.html'):
    """
    Save agent output to an HTML file.
    
    Args:
        result: The result object from the agent invocation
        filename: Output filename
    """
    html_output = format_agent_output_to_html(result)
    
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Sales Agent Output</title>
    </head>
    <body>
        {html_output}
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTML output saved to {filename}")
