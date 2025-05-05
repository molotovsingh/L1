import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Create database engine with connection pool settings
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # Add connection pool settings to help with temporary connection issues
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Test connections before using them
        pool_recycle=3600,   # Recycle connections after 1 hour
        pool_timeout=30,     # Wait up to 30 seconds for a connection
        max_overflow=10,     # Allow up to 10 extra connections when pool is full
        echo=False           # Don't log all SQL
    )
else:
    # Fallback for when no DATABASE_URL is provided
    import sqlite3
    print("WARNING: No DATABASE_URL found, using in-memory SQLite database")
    engine = create_engine("sqlite:///:memory:")

# Create declarative base
Base = declarative_base()

# Define models
class Taxonomy(Base):
    """Represents a domain taxonomy."""
    __tablename__ = "taxonomies"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    api_provider = Column(String(50), default="OpenAI")  # OpenAI or Perplexity
    tier_a_model = Column(String(50))
    tier_b_model = Column(String(50))
    max_labels = Column(Integer)
    min_labels = Column(Integer)
    deny_list = Column(Text)  # Stored as JSON
    
    # Relationships
    approved_labels = relationship("ApprovedLabel", back_populates="taxonomy", cascade="all, delete-orphan")
    rejected_labels = relationship("RejectedLabel", back_populates="taxonomy", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert taxonomy to dictionary representation."""
        return {
            "id": self.id,
            "domain": self.domain,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "api_provider": self.api_provider,
            "tier_a_model": self.tier_a_model,
            "tier_b_model": self.tier_b_model,
            "max_labels": self.max_labels,
            "min_labels": self.min_labels,
            "deny_list": json.loads(self.deny_list) if self.deny_list else [],
            "approved_labels": [label.label for label in self.approved_labels],
            "rejected_labels": {
                label.label: label.rejection_reason for label in self.rejected_labels
            }
        }


class ApprovedLabel(Base):
    """Represents an approved label in a taxonomy."""
    __tablename__ = "approved_labels"
    
    id = Column(Integer, primary_key=True)
    taxonomy_id = Column(Integer, ForeignKey("taxonomies.id"), nullable=False)
    label = Column(String(255), nullable=False)
    
    # Relationship
    taxonomy = relationship("Taxonomy", back_populates="approved_labels")


class RejectedLabel(Base):
    """Represents a rejected label in a taxonomy with a reason."""
    __tablename__ = "rejected_labels"
    
    id = Column(Integer, primary_key=True)
    taxonomy_id = Column(Integer, ForeignKey("taxonomies.id"), nullable=False)
    label = Column(String(255), nullable=False)
    rejection_reason = Column(Text)
    
    # Relationship
    taxonomy = relationship("Taxonomy", back_populates="rejected_labels")


# Create tables
Base.metadata.create_all(engine)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_taxonomy(domain, tier_a_model, tier_b_model, max_labels, min_labels, deny_list, 
                   approved_labels, rejected_labels, rejection_reasons, api_provider="OpenAI"):
    """
    Create a new taxonomy in the database.
    
    Args:
        domain (str): The domain for the taxonomy
        tier_a_model (str): The model used for Tier-A
        tier_b_model (str): The model used for Tier-B
        max_labels (int): Maximum number of labels
        min_labels (int): Minimum number of labels
        deny_list (set): Set of denied terms
        approved_labels (list): List of approved labels
        rejected_labels (list): List of rejected labels
        rejection_reasons (dict): Mapping of rejected labels to reasons
        api_provider (str): The API provider used ("OpenAI" or "Perplexity")
        
    Returns:
        int: The ID of the created taxonomy or None if there was an error
    """
    # First try to save to file system as a backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"taxonomies/{domain.replace(' ', '_')}_{timestamp}.json"
    try:
        os.makedirs("taxonomies", exist_ok=True)
        taxonomy_data = {
            "domain": domain,
            "timestamp": timestamp,
            "tier_a_model": tier_a_model,
            "tier_b_model": tier_b_model,
            "max_labels": max_labels,
            "min_labels": min_labels,
            "deny_list": list(deny_list),
            "approved_labels": approved_labels,
            "rejected_labels": {label: rejection_reasons.get(label, "No reason") for label in rejected_labels}
        }
        with open(filename, "w") as f:
            json.dump(taxonomy_data, f, indent=2)
        print(f"Taxonomy backed up to file: {filename}")
    except Exception as file_err:
        print(f"Warning: Failed to backup taxonomy to file: {file_err}")
    
    # Now try database save
    session = SessionLocal()
    try:
        # Create taxonomy
        taxonomy = Taxonomy(
            domain=domain,
            tier_a_model=tier_a_model,
            tier_b_model=tier_b_model,
            max_labels=max_labels,
            min_labels=min_labels,
            deny_list=json.dumps(list(deny_list))
        )
        session.add(taxonomy)
        session.flush()  # Get the ID before committing
        
        # Add approved labels
        for label in approved_labels:
            approved_label = ApprovedLabel(
                taxonomy_id=taxonomy.id,
                label=label
            )
            session.add(approved_label)
        
        # Add rejected labels with reasons
        for label in rejected_labels:
            reason = rejection_reasons.get(label, "No reason provided")
            rejected_label = RejectedLabel(
                taxonomy_id=taxonomy.id,
                label=label,
                rejection_reason=reason
            )
            session.add(rejected_label)
        
        session.commit()
        print(f"Taxonomy saved to database with ID: {taxonomy.id}")
        return taxonomy.id
    except Exception as e:
        session.rollback()
        
        # Log specific database error types for debugging
        error_msg = str(e).lower()
        if "ssl connection has been closed unexpectedly" in error_msg:
            print(f"Database connection error (SSL closed): {e}")
            print("This is likely a temporary connection issue. Taxonomy was saved to file as backup.")
        elif "connection timed out" in error_msg:
            print(f"Database connection timeout: {e}")
            print("The database connection timed out. Taxonomy was saved to file as backup.")
        elif "too many connections" in error_msg:
            print(f"Database connection pool error: {e}")
            print("Too many database connections. The server may be under heavy load.")
        else:
            print(f"Database error: {e}")
        
        # Return None to indicate failure, but don't crash - we have a file backup
        return None
    finally:
        session.close()


def get_taxonomies():
    """
    Get all taxonomies from the database.
    
    Returns:
        list: List of taxonomy dictionaries
    """
    session = SessionLocal()
    try:
        taxonomies = session.query(Taxonomy).all()
        return [taxonomy.to_dict() for taxonomy in taxonomies]
    finally:
        session.close()


def get_taxonomy(taxonomy_id):
    """
    Get a specific taxonomy by ID.
    
    Args:
        taxonomy_id (int): The ID of the taxonomy
        
    Returns:
        dict: The taxonomy as a dictionary or None if not found
    """
    session = SessionLocal()
    try:
        taxonomy = session.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()
        if taxonomy:
            return taxonomy.to_dict()
        return None
    finally:
        session.close()


def delete_taxonomy(taxonomy_id):
    """
    Delete a taxonomy by ID.
    
    Args:
        taxonomy_id (int): The ID of the taxonomy
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    session = SessionLocal()
    try:
        taxonomy = session.query(Taxonomy).filter(Taxonomy.id == taxonomy_id).first()
        if taxonomy:
            session.delete(taxonomy)
            session.commit()
            print(f"Taxonomy with ID {taxonomy_id} deleted successfully")
            return True
        print(f"Taxonomy with ID {taxonomy_id} not found")
        return False
    except Exception as e:
        session.rollback()
        
        # Log specific database error types for debugging
        error_msg = str(e).lower()
        if "ssl connection has been closed unexpectedly" in error_msg:
            print(f"Database connection error during delete (SSL closed): {e}")
            print("This is likely a temporary connection issue. Try again later.")
        elif "connection timed out" in error_msg:
            print(f"Database connection timeout during delete: {e}")
            print("The database connection timed out. Try again later.")
        else:
            print(f"Database error during delete: {e}")
        
        return False
    finally:
        session.close()