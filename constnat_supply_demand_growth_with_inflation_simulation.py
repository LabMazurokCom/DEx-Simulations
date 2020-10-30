import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import heapq

class TransactionType(Enum):
    ON_BOARDING = 1
    DRIVE_CREATION = 2
    CLIENT_PAY = 3
    DRIVE_PAY = 4

class Transaction:

    REPLICATORS_PER_DRIVE = 7

    def __init__(self, type: TransactionType):
        self.units = 20
        self.type = type
        self.xpx = -1
        self.fee = 0

    def to_units(self):
        if self.type == TransactionType.DRIVE_PAY:
            return False
        else:
            return True

class DEx:
    def __init__(self):
        self.xpx = 1000000
        self.units = 1000000
        self.fee = 0.003
        self.courses = []

    def get_liquidity(self):
        return self.xpx * self.units

    def fill_transaction(self, tx: Transaction):
        inflation_params = {'a': 0.01, 'b': 0.01}
        if tx.type == TransactionType.DRIVE_PAY:
            tx.units *= Transaction.REPLICATORS_PER_DRIVE
            tx.units = tx.units * (1 + inflation_params['b'])
        elif tx.type == TransactionType.CLIENT_PAY:
            tx.units *= Transaction.REPLICATORS_PER_DRIVE
            tx.units = tx.units * (1 - inflation_params['a'])
        elif tx.type == TransactionType.DRIVE_CREATION:
            tx.units *= Transaction.REPLICATORS_PER_DRIVE
        if tx.to_units():
            tx.xpx = (self.get_liquidity() / (self.units - tx.units) - self.xpx)
        else:
            units_after_fee = tx.units * (1 - self.fee)
            tx.xpx = self.xpx - (self.get_liquidity() / (self.units + units_after_fee))


    def get_course(self):
        return self.xpx / self.units

    def exchange(self, tx:Transaction):
        self.fill_transaction(tx)
        if tx.to_units():
            self.xpx += tx.xpx
            self.units -= tx.units
        else:
            self.xpx -= tx.xpx
            self.units += tx.units
        course = self.units / self.xpx
        self.xpx += tx.fee
        self.units += tx.fee * course
        if tx.type == TransactionType.ON_BOARDING:
            self.units += 4 * tx.units
        self.courses.append(self.get_course())


def generate_transactions(billing_periods, new_clients_per_billing_period, demand_supply_ratio,
                          billing_period_duration):
    new_replicators_per_billing_period = new_clients_per_billing_period / demand_supply_ratio \
                                         * Transaction.REPLICATORS_PER_DRIVE
    queue = []
    heapq.heappush(queue,
                   (np.random.exponential(billing_period_duration
                                                             / new_replicators_per_billing_period), Transaction(TransactionType.ON_BOARDING)))
    heapq.heappush(queue,
                   (np.random.exponential(billing_period_duration
                                                             / new_clients_per_billing_period), Transaction(TransactionType.DRIVE_CREATION)))
    replicators_awaiting = 0
    while (True):
        time, transaction = heapq.heappop(queue)
        if time > billing_periods * billing_period_duration:
            break
        if transaction.type == TransactionType.ON_BOARDING:
            heapq.heappush(queue, (time + np.random.exponential(billing_period_duration
                                                             / new_replicators_per_billing_period), Transaction(TransactionType.ON_BOARDING)))
            replicators_awaiting += 1
            yield transaction
        elif transaction.type == TransactionType.DRIVE_CREATION:
            heapq.heappush(queue, (time + np.random.exponential(billing_period_duration
                                                                / new_clients_per_billing_period),
                                   Transaction(TransactionType.DRIVE_CREATION)))
            if replicators_awaiting >= Transaction.REPLICATORS_PER_DRIVE:
                heapq.heappush(queue, (time + billing_period_duration,
                                       Transaction(TransactionType.DRIVE_PAY)))
                replicators_awaiting -= Transaction.REPLICATORS_PER_DRIVE
                yield transaction
        elif transaction.type == TransactionType.CLIENT_PAY:
            heapq.heappush(queue, (time + billing_period_duration,
                                   Transaction(TransactionType.DRIVE_PAY)))
            yield transaction
        elif transaction.type == TransactionType.DRIVE_PAY:
            heapq.heappush(queue, (time,
                                   Transaction(TransactionType.CLIENT_PAY)))
            yield transaction


np.random.seed(17)
exchanges = []
mean_courses = []
for i in range(1):
    print(i)
    try:
        exchange = DEx()
        for tx in generate_transactions(4, 1000, 1, 1):
            exchange.exchange(tx)
            print(tx.type)
        plt.plot(exchange.courses)
        plt.show()
    except:
        pass
    # plt.plot(np.linspace(0, 1, len(exchange.courses)), exchange.courses)
    # plt.show()