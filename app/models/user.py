import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum
from sqlalchemy.orm import relationship
from app.core.database import Base


class UserRole(str, enum.Enum):
    dataops_engineer = "DataOps Engineer"
    data_platform_lead = "Data Platform Lead"
    data_engineer = "Data Engineer"
    business_data_consumer = "Business Data Consumer"
    risk_officer = "Risk Officer"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    role = Column(
        Enum(UserRole, values_callable=lambda x: [e.value for e in x]),  # ← KEY FIX
        nullable=False,
        default=UserRole.data_engineer
    )

