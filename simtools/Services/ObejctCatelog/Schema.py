import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import PickleType
from sqlalchemy import String

from simtools.Services.ObejctCatelog import Base_object, engine_object


class Item(Base_object):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String)
    provider = Column(String)
    provider_info = Column(PickleType())
    item_id = Column(String)
    date_created = Column(DateTime(timezone=True), default=datetime.datetime.now())

    def endpoint(self):
        if self.provider_info:
            return self.provider_info.get("endpoint", None)

    def __repr__(self):
        return "Item (id=%s, provider=%s, provider_info=%s, item_id=%s)" % (self.id, self.provider, self.provider_info, self.item_id)


Base_object.metadata.create_all(engine_object)