#!/usr/bin/env python3
"""
Utility script to generate markdown documentation structure from YAML/JSON config files.

This script parses config files and generates markdown with the structure,
preserving comments and adding placeholders for manual descriptions.
"""
import fire
import json
import yaml
from pathlib import Path
from typing import Any, Dict


def extract_yaml_comments(file_path: Path) -> Dict[str, str]:
    """Extract comments from YAML file."""
    comments = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        current_key = None
        key_path = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if line is a comment
            if stripped.startswith('#'):
                comment = stripped[1:].strip()
                if current_key:
                    comments[' -> '.join(key_path)] = comment
            # Check if line has a key
            elif ':' in stripped and not stripped.startswith('#'):
                key = stripped.split(':')[0].strip()
                # Determine nesting level
                indent = len(line) - len(line.lstrip())
                # Adjust key_path based on indent
                while len(key_path) > 0 and indent <= (len(key_path) - 1) * 2:
                    key_path.pop()
                key_path.append(key)
                current_key = key
    
    return comments


def format_value(value: Any, indent: int = 0) -> str:
    """Format a config value for markdown display."""
    indent_str = "  " * indent
    
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{indent_str}{k}:")
                lines.append(format_value(v, indent + 1))
            else:
                lines.append(f"{indent_str}{k}: {format_single_value(v)}")
        return "\n".join(lines)
    elif isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{indent_str}-")
                lines.append(format_value(item, indent + 1))
            else:
                lines.append(f"{indent_str}- {format_single_value(item)}")
        return "\n".join(lines)
    else:
        return f"{indent_str}{format_single_value(value)}"


def format_single_value(value: Any) -> str:
    """Format a single value."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        # Check if it needs quotes
        if any(char in value for char in [' ', ':', '#', '|', '&', '*', '?', '[', ']', '{', '}']):
            return f'"{value}"'
        return value
    else:
        return str(value)


def generate_config_doc(config_path: Path, output_path: Path, title: str, description: str = None) -> None:
    """
    Generate markdown documentation from a config file.
    
    Args:
        config_path: Path to the config file (YAML or JSON)
        output_path: Path where markdown should be written
        title: Title for the documentation
        description: Optional description to add
    """
    # Read config file
    if config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        comments = {}
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        comments = extract_yaml_comments(config_path)
    
    # Generate markdown
    lines = [
        f"# {title}",
        "",
        "> **Location**: `{config_path}`",
        "",
    ]
    
    if description:
        lines.extend([
            description,
            "",
        ])
    else:
        lines.extend([
            "> **TODO**: Add description of what this configuration file is used for.",
            "",
        ])
    
    lines.extend([
        "## Configuration Structure",
        "",
        "### Complete Parameter Reference",
        "",
        "```yaml",
    ])
    
    # Add the YAML structure
    yaml_str = yaml.dump(config_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    lines.append(yaml_str)
    lines.append("```")
    
    # Add parameter descriptions section
    lines.extend([
        "",
        "### Parameter Descriptions",
        "",
        "> **TODO**: Add descriptions for each parameter.",
        "",
    ])
    
    # Add example configurations
    lines.extend([
        "---",
        "",
        "## Example Configurations",
        "",
        "> **TODO**: Add example configurations for common use cases.",
        "",
    ])
    
    # Add best practices
    lines.extend([
        "---",
        "",
        "## Best Practices",
        "",
        "> **TODO**: Add best practices and usage tips.",
        "",
    ])
    
    # Add troubleshooting
    lines.extend([
        "---",
        "",
        "## Troubleshooting",
        "",
        "> **TODO**: Add troubleshooting tips for common issues.",
        "",
    ])
    
    # Add navigation links
    lines.extend([
        "---",
        "",
        "## Related Documentation",
        "",
        "- [Configuration Overview](../index.md)",
        "- [Scripts Reference](../../scripts/wildetect/index.md)",
        "",
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    fire.Fire(generate_config_doc)
