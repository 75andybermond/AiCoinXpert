"""Basic database setup."""
import sqlalchemy as _sql
import sqlalchemy.ext.declarative as _declarative
import sqlalchemy.orm as _orm

DATABASE_URL = "postgresql://root:password@localhost:5432/coco"

## connect to the database
engine = _sql.create_engine(DATABASE_URL)

## create a session
SessionLocal = _orm.sessionmaker(autocommit=False, autoflush=False, bind=engine)

## create a base class for our models to inherit from
Base = _declarative.declarative_base()
