from app.models import Base
from app import engine

Base.metadata.create_all(bind=engine)