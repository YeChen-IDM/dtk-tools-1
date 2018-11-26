import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

current_dir = os.path.dirname(os.path.realpath(__file__))

# General Metadata DB
engine_object = create_engine('sqlite:///%s/object.sqlite' % current_dir, echo=False, connect_args={'timeout': 90})
Session = sessionmaker(bind=engine_object)
Base_object = declarative_base()

@contextmanager
def session_scope(session=None):

    """Provide a transactional scope around a series of operations."""
    session = Session() if not session else session
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


