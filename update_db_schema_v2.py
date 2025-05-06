"""
Update database schema to add raw output and timestamp columns to taxonomies table
"""

import os
import logging
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get database URL
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set")
    exit(1)

def update_taxonomy_schema():
    """Add raw output and timestamp columns to taxonomies table if they don't exist"""
    # Connect to database
    engine = create_engine(DATABASE_URL)
    
    # Create a new connection for each operation to avoid transaction issues
    def run_with_new_connection(sql, description):
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
                logger.info(f"Successfully {description}")
                return True
        except Exception as e:
            logger.error(f"Error {description}: {e}")
            return False
    
    # Define new columns to add
    columns_to_add = [
        {
            "name": "tier_a_raw_output",
            "type": "TEXT",
            "description": "raw output from Tier-A model"
        },
        {
            "name": "tier_b_raw_output",
            "type": "TEXT",
            "description": "raw output from Tier-B model"
        },
        {
            "name": "tier_a_timestamp",
            "type": "TIMESTAMP",
            "default": "NULL",
            "description": "timestamp of Tier-A API call"
        },
        {
            "name": "tier_b_timestamp",
            "type": "TIMESTAMP",
            "default": "NULL",
            "description": "timestamp of Tier-B API call"
        }
    ]
    
    # Check and add each column
    for column in columns_to_add:
        try:
            with engine.connect() as conn:
                # Check if column exists
                result = conn.execute(text(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'taxonomies' AND column_name = '{column["name"]}'
                """))
                if result.fetchone():
                    logger.info(f"{column['name']} column already exists")
                else:
                    # Column doesn't exist, add it
                    logger.info(f"{column['name']} column does not exist, adding it now")
                    
                    # Build ALTER TABLE statement
                    default_clause = f"DEFAULT {column['default']}" if "default" in column else ""
                    alter_sql = f"""
                    ALTER TABLE taxonomies 
                    ADD COLUMN {column["name"]} {column["type"]} {default_clause}
                    """
                    
                    # Execute the ALTER TABLE statement
                    success = run_with_new_connection(alter_sql, f"added {column['name']} column to taxonomies table")
                    if not success:
                        logger.error(f"Failed to add {column['name']} column")
        except Exception as e:
            logger.error(f"Error checking/adding {column['name']} column: {e}")
    
    # Rename max_labels/min_labels columns (optional approach, but we'll skip for now and handle in app logic)
    # This would break existing data access, so we'll just keep the original names
    
    # Cleanup
    engine.dispose()

if __name__ == "__main__":
    update_taxonomy_schema()