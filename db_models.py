import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Create database engine
DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Create declarative base
Base = declarative_base()

# Define models
class Taxonomy(Base):
    """Represents a domain taxonomy."""
    __tablename__ = "taxonomies"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
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
                   approved_labels, rejected_labels, rejection_reasons):
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
        
    Returns:
        int: The ID of the created taxonomy
    """
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
        return taxonomy.id
    except Exception as e:
        session.rollback()
        raise e
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
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()