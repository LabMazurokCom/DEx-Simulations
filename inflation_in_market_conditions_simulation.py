import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import heapq


class Transaction:

    def __init__(self, factor):
        self.fee = 0.03
        self.units = 20 * factor
        self.xpx = -1

    def to_units(self):
        pass


class ReplicatorBoardingTransaction(Transaction):
    def __init__(self):
        Transaction.__init__(self, 1)

    def to_units(self):
        pass

    def __str__(self):
        return 'ReplicatorBoardingTransaction'

class ReplicatorOffBoardingTransaction(ReplicatorBoardingTransaction):
    def __init__(self):
        Transaction.__init__(self, 1)

    def to_units(self):
        return False

    def __str__(self):
        return 'ReplicatorOffBoardingTransaction'


class ReplicatorOnBoardingTransaction(ReplicatorBoardingTransaction):
    def __init__(self):
        Transaction.__init__(self, 1)

    def to_units(self):
        return True

    def __str__(self):
        return 'ReplicatorOnBoardingTransaction'


class DriveCreationTransaction(Transaction):
    MAX_REPLICATORS_PER_DRIVE = 7

    def __init__(self):
        Transaction.__init__(self, DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)

    def to_units(self):
        return True

    def __str__(self):
        return 'DriveCreationTransaction'


class DriveEndTransaction(Transaction):
    def __init__(self, factor):
        Transaction.__init__(self, factor)

    def to_units(self):
        return False

    def __str__(self):
        return 'DriveEndTransaction'


class PayToReplicatorsTransaction(Transaction):

    def __init__(self, factor):
        Transaction.__init__(self,factor)
        self.factor = factor

    def to_units(self):
        return False

    def specify_replicators(self, factor):
        self.factor = (self.factor + factor) / 2
        Transaction.__init__(self, self.factor)

    def get_replicators(self):
        return self.factor

    def __str__(self):
        return 'PayToReplicatorsTransaction'


class DriveProlongationTransaction(Transaction):

    def __init__(self, factor):
        Transaction.__init__(self, factor)

    def to_units(self):
        return True

    def __str__(self):
        return 'DriveProlongationTransaction'


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
        transaction_type = type(tx)
        if transaction_type == ReplicatorOnBoardingTransaction:
            pass
        elif transaction_type == ReplicatorOffBoardingTransaction:
            pass
        elif transaction_type == DriveCreationTransaction:
            pass
        elif transaction_type == DriveProlongationTransaction:
            tx.units = tx.units * (1 - inflation_params['a'])
        elif transaction_type == PayToReplicatorsTransaction:
            tx.units = tx.units / (1 - inflation_params['a']) * (1 + inflation_params['b'])
        elif transaction_type == DriveEndTransaction:
            pass
        if tx.to_units():
            tx.xpx = (self.get_liquidity() / (self.units - tx.units) - self.xpx)
        else:
            units_after_fee = tx.units * (1 - self.fee)
            tx.xpx = self.xpx - (self.get_liquidity() / (self.units + units_after_fee))

    def get_course(self):
        return self.xpx / self.units

    def after_exchange_process(self, tx: Transaction):
        transaction_type = type(tx)
        if transaction_type == ReplicatorOnBoardingTransaction:
            self.units += 4 * tx.units
        elif transaction_type == ReplicatorOffBoardingTransaction:
            self.units -= 4 * tx.units

    def exchange(self, tx: Transaction):
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
        self.after_exchange_process(tx)
        self.courses.append(self.get_course())


def get_flow(course):
    return 2 * 0.75 * 3 ** course / (1 + 0.75 * (3 ** course - 1)) - 1


def perform_action(probability):
    return np.random.choice([True, False], p=[probability, 1 - probability])


class TransactionsGenerator:
    def __init__(self, init_course):
        self.init_course = init_course
        self.course = init_course

    def notify_course_changed(self, course):
        self.course = course

    def get_log_course_change(self):
        return np.log2(self.course / self.init_course)

    def generate_transactions(self, billing_periods, max_new_clients_per_billing_period, demand_supply_ratio,
                              billing_period_duration):
        max_new_replicators_per_billing_period = \
            max_new_clients_per_billing_period / demand_supply_ratio \
            * DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE
        queue = []
        heapq.heappush(queue,
                       (np.random.exponential(billing_period_duration
                                                                 / max_new_replicators_per_billing_period), ReplicatorBoardingTransaction()))
        heapq.heappush(queue,
                       (np.random.exponential(billing_period_duration
                                              / max_new_clients_per_billing_period), DriveCreationTransaction()))
        active_replicators = 0
        active_drives = 0
        while True:
            time, transaction = heapq.heappop(queue)
            if time > billing_periods * billing_period_duration:
                break
            replicators_flow = get_flow(self.get_log_course_change())
            clients_flow = get_flow(-self.get_log_course_change())
            transaction_type = type(transaction)
            if transaction_type == ReplicatorBoardingTransaction:
                heapq.heappush(queue, (time + np.random.exponential(billing_period_duration
                                                                 / max_new_replicators_per_billing_period), ReplicatorBoardingTransaction()))
                make_transaction = perform_action(np.abs(replicators_flow))
                if make_transaction:
                    if replicators_flow > 0:
                        active_replicators += 1
                        yield time, ReplicatorOnBoardingTransaction()
                    else:
                        active_replicators -= 1
                        yield time, ReplicatorOffBoardingTransaction()
            elif transaction_type == DriveCreationTransaction:
                heapq.heappush(queue, (time + np.random.exponential(billing_period_duration
                                                                    / max_new_clients_per_billing_period),
                                       DriveCreationTransaction()))
                if clients_flow > 0:
                    make_transaction = perform_action(clients_flow)
                    if make_transaction:
                        active_drives += 1
                        replicators = min(active_replicators / (active_drives + 1),
                                          DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)
                        heapq.heappush(queue, (time + billing_period_duration,
                                               PayToReplicatorsTransaction(replicators)))
                        yield time, transaction
            elif transaction_type == PayToReplicatorsTransaction:
                replicators = min(active_replicators / active_drives,
                                  DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)
                transaction.specify_replicators(replicators)
                prolong = clients_flow >= 0
                if clients_flow < 0:
                    proba = min(1, -clients_flow * max_new_clients_per_billing_period / active_drives)
                    prolong = perform_action(proba)
                if prolong:
                    heapq.heappush(queue, (time, DriveProlongationTransaction(
                        transaction.get_replicators())))
                else:
                    active_drives -= 1
                    heapq.heappush(queue, (time, DriveEndTransaction(
                        DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE -
                        transaction.get_replicators())))
                yield time, transaction
            elif transaction_type == DriveProlongationTransaction:
                replicators = min(active_replicators / active_drives,
                                  DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)
                heapq.heappush(queue, (time + billing_period_duration,
                                       PayToReplicatorsTransaction(replicators)))
                yield time, transaction
            elif transaction_type == ReplicatorOffBoardingTransaction:
                yield time, transaction
            elif transaction_type == DriveEndTransaction:
                yield time, transaction
        print('Replicators Flow', replicators_flow)
        print('Clients Flow', clients_flow)


def add_transactions_to_statistics(tx, statistics):
    tx_name = str(tx)
    if tx_name not in statistics:
        statistics[tx_name] = 0
    statistics[tx_name] += 1


np.random.seed(17)
exchange = DEx()
transactions = dict()
times = []
transactions_generator = TransactionsGenerator(exchange.get_course())
for time, tx in transactions_generator.generate_transactions(200, 2000, 1, 1):
    times.append(time)
    if type(tx) == PayToReplicatorsTransaction or type(tx) == DriveCreationTransaction:
        print(tx)
    if type(tx) == PayToReplicatorsTransaction:
        print(tx.factor, tx.units, exchange.get_course(), time)
    exchange.exchange(tx)
    add_transactions_to_statistics(tx, transactions)
    transactions_generator.notify_course_changed(exchange.get_course())
    # print(type(tx))
plt.plot(times, exchange.courses)
plt.show()