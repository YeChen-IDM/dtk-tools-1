import datetime

from simtools.DataAccess import session_scope
from simtools.DataAccess.Schema import Simulation
from sqlalchemy import and_
from sqlalchemy import bindparam
from sqlalchemy import not_
from sqlalchemy import update

from simtools.Utilities.General import init_logging, batch, batch_list
from sqlalchemy.orm import joinedload

from COMPS.Data.Simulation import SimulationState
logger = init_logging('DataAccess')


class SimulationDataStore:

    @classmethod
    def batch_simulations_update(cls, simulation_batch):
        """
        Takes a batch of simulations and update their status in the DB.
        This function provides performance considerations when updating large number of simulations in the db.

        The batch needs to be formatted as follow:
        [
            {'sid':'simid', "status": 'simstatus'},
            {'sid':'simid', "status": 'simstatus'}
        ]

        Args:
            batch: Batch of simulations to save
        """
        if len(simulation_batch) == 0: return

        with session_scope() as session:
            stmt = update(Simulation).where(and_(Simulation.id == bindparam("sid"),
                                                 not_(Simulation.status in (
                                                 SimulationState.Succeeded, SimulationState.Failed,
                                                 SimulationState.Canceled)))) \
                .values(status_s=bindparam("status"))
            for sim_batch in batch_list(simulation_batch, 2500):
                session.execute(stmt, sim_batch)

    @classmethod
    def bulk_insert_simulations(cls, simulations):
        with session_scope() as session:
            for sim_batch in batch_list(simulations, 2500):
                session.bulk_save_objects(sim_batch)

    @classmethod
    def create_simulation(cls, **kwargs):
        if 'date_created' not in kwargs:
            kwargs['date_created'] = datetime.datetime.now()
        return Simulation(**kwargs)

    @classmethod
    def save_simulation(cls, simulation, session=None):
        with session_scope(session) as session:
            session.merge(simulation)

    @classmethod
    def get_simulation(cls, sim_id):
        with session_scope() as session:
            simulation = session.query(Simulation).options(joinedload('experiment')).filter(Simulation.id == sim_id).one_or_none()
            session.expunge_all()

        return simulation

    @classmethod
    def delete_simulation(cls, simulation):
        with session_scope() as session:
            session.query(Simulation).filter(Simulation.id == simulation.id).delete(synchronize_session=False)
