from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.sql import func

Base = declarative_base()

class Round(Base):
    __tablename__ = 'rounds'

    id = Column('id', Integer, primary_key=True, autoincrement=True)
    timestamp = Column('timestamp', DateTime, server_default=func.now())
    estimator = Column('estimator', String, nullable=False)
    params = Column('params', String, nullable=False)
    acc = Column('acc', Float, default=0.0)
    macro_pre = Column('macro_pre', Float, default=0.0)
    macro_rec = Column('macro_rec', Float, default=0.0)
    macro_f1 = Column('macro_f1', Float, default=0.0)
    micro_f1 = Column('micro_f1', Float, default=0.0)

    def __init__(self, estimator, params, acc,
                 macro_pre, macro_rec, macro_f1,
                 micro_f1):
        self.estimator = estimator
        self.params = params
        self.acc = acc
        self.macro_pre = macro_pre
        self.macro_rec = macro_rec
        self.macro_f1 = macro_f1
        self.micro_f1 = micro_f1