#!/usr/bin/env python3
"""
process_data_graph4.py - Transform concept_carina4.xlsx into Neo4j-ready format
Handles the streamlined A/B structure: Category A, Value A, Relationship, Value B, Category B
"""

import pandas as pd
import json
import os
import re
import uuid
import argparse

def clean_text(text):
    """Clean and normalize text values"""
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip()
    return text if text else None

def generate_id(text, prefix='concept'):
    """Generate a unique ID based on text"""
    if not text or pd.isna(text):
        return f"{prefix}_{str(uuid.uuid4())[:8]}"
    
    slug = re.sub(r'[^a-zA-Z0-9]', '_', text.lower())
    slug = re.sub(r'_+', '_', slug)[:30]
    return f"{prefix}_{slug}"

def category_to_label(category):
    """Convert a category string to a valid Neo4j label"""
    if not category or pd.isna(category):
        return "Concept"
    
    # Handle special cases
    special_cases = {
        "student with special needs": "StudentWithSpecialNeeds",
        "learning strategy": "LearningStrategy", 
        "pedagogical methodology": "PedagogicalMethodology",
        "learner profile": "LearnerProfile",
        "class climate": "ClassClimate",
        "learning setting": "LearningSetting",
        "teaching approach": "TeachingApproach",
        "technologies": "Technology",
        "learning process": "LearningProcess",
        "environmental psychology": "EnvironmentalPsychology",
        "pedagogical approach": "PedagogicalApproach",
        "pedagogical strategy": "PedagogicalStrategy"
    }
    
    category_lower = category.lower().strip()
    if category_lower in special_cases:
        return special_cases[category_lower]
    
    # Convert to CamelCase
    words = re.sub(r'[^a-zA-Z0-9\s]', '', category).split()
    label = ''.join(word.capitalize() for word in words)
    
    # Make singular
    if label.endswith('s') and not label.endswith('ss') and len(label) > 3:
        label = label[:-1]
    
    return label if label else "Concept"

def process_excel_data(input_file='concept_carina4.xlsx', output_file='concepts4_neo4j.json'):
    """Process Excel file with streamlined A/B structure into Neo4j-ready format"""

    # Load Excel file
    print(f"Loading {input_file}...")
    df = pd.read_excel(input_file)

    # Handle different column naming conventions
    column_mappings = {
        # Old standard format
        'Value A': 'Value A',
        'Relationship': 'Relationship',
        'Value B': 'Value B',
        'Category A': 'Category A',
        'Category B': 'Category B',
        # New neuroscience format
        'Value A (nodo 1)': 'Value A',
        'Relationship (freccia)': 'Relationship',
        'Value B (nodo 2/di arrivo)': 'Value B',
        'Concept A': 'Category A',  # Map Concept A to Category A
        'Concept B': 'Category B',  # Map Concept B to Category B
    }

    # Map columns to expected names
    df_mapped = df.copy()
    for old_col, standard_col in column_mappings.items():
        if old_col in df.columns and standard_col not in df.columns:
            df_mapped = df_mapped.rename(columns={old_col: standard_col})

    # Validate columns - now expecting only 5 essential columns
    expected_cols = ['Category A', 'Value A', 'Relationship', 'Value B', 'Category B']
    missing_cols = [col for col in expected_cols if col not in df_mapped.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        print("Column mapping attempted:")
        for old, standard in column_mappings.items():
            if old in df.columns:
                print(f"  '{old}' → '{standard}'")
        return False

    print(f"✅ Found all required columns: {expected_cols}")
    print(f"Processing {len(df)} rows...")
    print(f"Using neuroscience knowledge graph format")

    # Initialize output structure
    processed_data = {"nodes": [], "relationships": []}
    node_registry = {}  # Track unique nodes
    
    valid_rows = 0
    skipped_rows = []
    
    for index, row in df.iterrows():
        # Extract values from the 5 essential columns
        value_a = clean_text(row.get('Value A'))
        value_b = clean_text(row.get('Value B'))
        category_a = clean_text(row.get('Category A'))
        category_b = clean_text(row.get('Category B'))
        relationship = clean_text(row.get('Relationship'))
        
        # Skip invalid rows and track them
        missing_fields = []
        if not value_a: missing_fields.append("Value A")
        if not value_b: missing_fields.append("Value B") 
        if not category_a: missing_fields.append("Category A")
        if not category_b: missing_fields.append("Category B")
        if not relationship: missing_fields.append("Relationship")
        
        if missing_fields:
            skipped_rows.append(f"Row {index + 1}: missing {', '.join(missing_fields)}")
            continue
        
        # Convert categories to Neo4j labels
        label_a = category_to_label(category_a)
        label_b = category_to_label(category_b)
        
        # Create node A if not exists
        if value_a not in node_registry:
            node_a_id = generate_id(value_a)
            node_registry[value_a] = {'id': node_a_id, 'label': label_a}
            
            processed_data["nodes"].append({
                "label": label_a,
                "properties": {
                    "id": node_a_id,
                    "name": value_a,
                    "category": category_a
                }
            })
        
        # Create node B if not exists
        if value_b not in node_registry:
            node_b_id = generate_id(value_b)
            node_registry[value_b] = {'id': node_b_id, 'label': label_b}
            
            processed_data["nodes"].append({
                "label": label_b,
                "properties": {
                    "id": node_b_id,
                    "name": value_b,
                    "category": category_b
                }
            })
        
        # Create relationship
        rel_type = relationship.upper().replace(' ', '_')
        processed_data["relationships"].append({
            "from": node_registry[value_a]['id'],
            "to": node_registry[value_b]['id'],
            "type": rel_type
        })
        
        valid_rows += 1
    
    # Report any skipped rows
    if skipped_rows:
        print(f"\n⚠️  Skipped {len(skipped_rows)} rows with missing data:")
        for skip_msg in skipped_rows[:5]:  # Show first 5
            print(f"   {skip_msg}")
        if len(skipped_rows) > 5:
            print(f"   ... and {len(skipped_rows) - 5} more")
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, indent=2, ensure_ascii=False)
    
    # Print comprehensive summary
    print(f"\n✅ Processing complete!")
    print(f"  📊 Valid rows processed: {valid_rows}/{len(df)}")
    print(f"  🎯 Unique nodes created: {len(processed_data['nodes'])}")
    print(f"  🔗 Relationships created: {len(processed_data['relationships'])}")
    print(f"  💾 Output saved to: {output_file}")
    
    # Node label distribution
    label_counts = {}
    for node in processed_data["nodes"]:
        label = node["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n📋 Node label distribution ({len(label_counts)} unique labels):")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count} nodes")
    
    # Relationship type distribution
    rel_counts = {}
    for rel in processed_data["relationships"]:
        rel_type = rel["type"]
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    print(f"\n🔗 Relationship type distribution ({len(rel_counts)} unique types):")
    for rel_type, count in sorted(rel_counts.items()):
        print(f"   {rel_type}: {count} relationships")
    
    return True

def fix_property_names_in_json(json_file='kg_neuro_neo4j.json', backup=True):
    """Fix property names in existing JSON file: change 'concept' to 'category'"""

    print(f"🔧 Fixing property names in {json_file}...")

    # Load existing JSON
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return False

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Backup original if requested
    if backup:
        backup_file = json_file.replace('.json', '_backup.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"📋 Backup created: {backup_file}")

    # Count changes needed
    changed_nodes = 0
    total_nodes = len(data.get('nodes', []))

    # Fix each node
    for node in data.get('nodes', []):
        properties = node.get('properties', {})

        # If node has 'concept' property, rename it to 'category'
        if 'concept' in properties and 'category' not in properties:
            properties['category'] = properties.pop('concept')
            changed_nodes += 1

        # Update the node properties
        node['properties'] = properties

    # Save fixed JSON
    fixed_file = json_file.replace('.json', '_fixed.json')
    with open(fixed_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Fixed {changed_nodes}/{total_nodes} nodes")
    print(f"💾 Fixed file saved as: {fixed_file}")

    # If all nodes were changed, replace original file
    if changed_nodes == total_nodes and changed_nodes > 0:
        original_file = json_file.replace('.json', '_original.json')
        os.rename(json_file, original_file)
        os.rename(fixed_file, json_file)
        print(f"📝 Original file backed up as: {original_file}")
        print(f"🔄 Fixed file is now: {json_file}")
    elif changed_nodes > 0:
        print(f"⁉️  Only {changed_nodes} nodes needed fixing - kept original")
    else:
        print(f"ℹ️  No 'concept' properties found - all nodes already correct")

    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Transform Excel to Neo4j JSON or fix existing JSON property names',
        epilog='Use --fix to fix property names in existing JSON files'
    )

    # Choose between process or fix modes
    parser.add_argument('--mode', choices=['process', 'fix'], default='process',
                       help='Mode: process Excel -> JSON or fix existing JSON property names')

    # Excel processing arguments
    parser.add_argument('--input', default='concept_carina4.xlsx', help='Input Excel file (for process mode)')
    parser.add_argument('--output', default='concepts4_neo4j.json', help='Output JSON file (for process mode)')

    # JSON fixing arguments
    parser.add_argument('--fix', default='kg_neuro_neo4j.json', help='JSON file to fix (for fix mode)')
    parser.add_argument('--no-backup', action='store_false', dest='backup',
                       help='Skip creating backup of original file (dangerous!)')

    args = parser.parse_args()

    if args.mode == 'process':
        # Excel processing mode
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            return

        print("🚀 Starting Excel to Neo4j transformation...")
        print(f"📁 Input: {args.input}")
        print(f"💾 Output: {args.output}")

        success = process_excel_data(args.input, args.output)

        if success:
            print(f"\n🎉 Ready to import into Neo4j:")
            print(f"   python data_ingestion_neo4j.py --file {args.output} --password YOUR_PASSWORD --clear")
            print(f"\n💡 This structure is optimized for GraphRAG text2Cypher queries!")
        else:
            print("❌ Processing failed")

    elif args.mode == 'fix':
        # JSON fixing mode
        print("🛠️  Starting JSON property name fix...")
        print(f"📁 Target: {args.fix}")
        print(f"📋 Backup: {'enabled' if args.backup else 'disabled - DANGER!'}")

        success = fix_property_names_in_json(args.fix, args.backup)

        if success:
            print("\n✅ Property fix complete!")
            print("🔧 'concept' → 'category' conversion finished")
        else:
            print("❌ Property fix failed")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
