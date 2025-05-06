#!/usr/bin/env python3
"""
Update Database Schema v2 - Add Custom Prompt Fields to Taxonomy Table

This script adds tier_a_prompt_id and tier_b_prompt_id columns to the taxonomies table
to store references to custom prompts used for generation.

Run this script to update the database schema.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.sql import table, column
from sqlalchemy import Integer, String, Text, DateTime, MetaData, Table, Column
import sqlalchemy.exc

def update_schema():
    """Update the database schema to add custom prompt ID fields."""
    
    # Get database URL
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is not set")
        return False
    
    # Create engine
    try:
        engine = create_engine(db_url)
        conn = engine.connect()
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return False
    
    try:
        # Define the new columns we need to add
        metadata = MetaData()
        taxonomy_table = Table('taxonomies', metadata, autoload_with=engine)
        
        # Check if the columns already exist
        existing_columns = [c.name for c in taxonomy_table.columns]
        
        # Add tier_a_prompt_id column if it doesn't exist
        if 'tier_a_prompt_id' not in existing_columns:
            conn.execute(text(
                "ALTER TABLE taxonomies ADD COLUMN tier_a_prompt_id INTEGER"
            ))
            print("Added tier_a_prompt_id column to taxonomies table")

        # Add tier_b_prompt_id column if it doesn't exist
        if 'tier_b_prompt_id' not in existing_columns:
            conn.execute(text(
                "ALTER TABLE taxonomies ADD COLUMN tier_b_prompt_id INTEGER"
            ))
            print("Added tier_b_prompt_id column to taxonomies table")
        
        # Commit changes
        conn.commit()
        print("Schema update v2 completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to update schema: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    """Main entry point."""
    print("Starting schema update v2...")
    if update_schema():
        print("Schema update v2 successful")
    else:
        print("Schema update v2 failed")
        sys.exit(1)

if __name__ == "__main__":
    main()