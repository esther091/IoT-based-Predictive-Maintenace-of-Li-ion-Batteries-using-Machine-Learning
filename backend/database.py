"""
database.py
SQLite database setup using SQLAlchemy.
Defines the three tables: readings, cycles, predictions.
"""

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./battery.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Reading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    power = Column(Float, nullable=False)
    energy = Column(Float, nullable=False)


class Cycle(Base):
    __tablename__ = "cycles"

    id = Column(Integer, primary_key=True, index=True)
    cycle_number = Column(Integer)
    cycle_start = Column(DateTime)
    cycle_end = Column(DateTime)
    mean_voltage = Column(Float)
    mean_current = Column(Float)
    mean_power = Column(Float)
    energy_delta = Column(Float)
    capacity_mah = Column(Float)        # charge delivered, integral of |I| dt
    min_cell_voltage = Column(Float)    # lowest per-cell voltage under load
    complete = Column(Integer)          # 1 = ran to cut-off, 0 = partial


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    cycle_number = Column(Integer)
    soh_rf = Column(Float)
    soh_xgb = Column(Float)
    soh_final = Column(Float)
    rul_rf = Column(Float)
    rul_xgb = Column(Float)
    rul_final = Column(Float)
    alert_level = Column(String)


def create_tables():
    Base.metadata.create_all(bind=engine)
    # create_all never alters an existing table, so add columns introduced
    # after a battery.db was first created
    with engine.begin() as conn:
        existing = {c["name"] for c in inspect(conn).get_columns("cycles")}
        for name, sqltype in (("capacity_mah", "FLOAT"), ("min_cell_voltage", "FLOAT"),
                              ("complete", "INTEGER")):
            if name not in existing:
                conn.execute(text(f"ALTER TABLE cycles ADD COLUMN {name} {sqltype}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()