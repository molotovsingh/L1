"""
Update database schema to add api_provider column to taxonomies table
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

def add_api_provider_column():
    """Add api_provider column to taxonomies table if it doesn't exist"""
    # Connect to database with autocommit=True to avoid transaction issues
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
    
    # Check if column exists by describing the table
    try:
        with engine.connect() as conn:
            # Get column information from information_schema to avoid errors
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'taxonomies' AND column_name = 'api_provider'
            """))
            if result.fetchone():
                logger.info("api_provider column already exists")
                return
            else:
                logger.info("api_provider column does not exist, adding it now")
    except Exception as e:
        logger.error(f"Error checking column existence: {e}")
        return
    
    # Add column to taxonomies table
    alter_sql = """
    ALTER TABLE taxonomies 
    ADD COLUMN api_provider VARCHAR(50) DEFAULT 'OpenAI'
    """
    run_with_new_connection(alter_sql, "added api_provider column to taxonomies table")
    
    # Cleanup
    engine.dispose()

if __name__ == "__main__":
    add_api_provider_column()