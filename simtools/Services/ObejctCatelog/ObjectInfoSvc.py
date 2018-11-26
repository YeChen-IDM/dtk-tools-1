import datetime
from operator import or_

from simtools.Services.ObejctCatelog import session_scope
from simtools.Services.ObejctCatelog.Schema import Item


class ObjectInfoSvc:

    @staticmethod
    def get_item_info(id_or_item_id):
        with session_scope() as session:
            item = session.query(Item).filter(or_(Item.id == id_or_item_id, Item.item_id == id_or_item_id)).one_or_none()

            if item:
                return {"item_id": item.item_id, "type": item.type, "provider": item.provider, "provider_info": item.provider_info}


    @staticmethod
    def get_item(id_or_item_id):
        # print(id_or_item_id, ' : ', type(id_or_item_id))
        with session_scope() as session:
            item = session.query(Item).filter(or_(Item.id == id_or_item_id, Item.item_id == id_or_item_id)).one_or_none()
            session.expunge_all()

        return item

    @classmethod
    def create_item(cls, **kwargs):
        if 'date_created' not in kwargs:
            kwargs['date_created'] = datetime.datetime.now()

        item = Item(**kwargs)
        with session_scope() as session:
            if not item.id:
                session.add(item)
            else:
                session.merge(item)


