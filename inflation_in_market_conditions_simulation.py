import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import heapq
from collections import deque


class Transaction:

    def __init__(self, factor):
        self.units = 20 * factor
        self.xpx = -1
        self.fee = 0

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
        # real_factor = DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE * revenue_rate
        # # real_factor = DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE
        Transaction.__init__(self, DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)

    def to_units(self):
        return True

    def __str__(self):
        return 'DriveCreationTransaction'

    def specify_params(self, revenue_rate):
        real_factor = DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE * revenue_rate
        Transaction.__init__(self, real_factor)


class DriveEndTransaction(Transaction):
    def __init__(self, factor):
        Transaction.__init__(self, factor)

    def to_units(self):
        return False

    def __str__(self):
        return 'DriveEndTransaction'


class PayToReplicatorsTransaction(Transaction):

    def __init__(self, replicators, revenue_rate):
        real_factor = replicators * revenue_rate
        Transaction.__init__(self, real_factor)
        self.initial_replicators = replicators
        self.finish_replicators = replicators
        self.revenue_rate = revenue_rate

    def to_units(self):
        return False

    def specify_params(self, replicators):
        self.finish_replicators = replicators
        replicators = (self.initial_replicators + self.finish_replicators) / 2
        real_factor = replicators * self.revenue_rate
        Transaction.__init__(self, real_factor)

    def get_replicators(self):
        return (self.initial_replicators + self.finish_replicators) / 2

    def __str__(self):
        return 'PayToReplicatorsTransaction'


class DriveProlongationTransaction(Transaction):

    def __init__(self, factor):
        factor = max(factor, 0)
        Transaction.__init__(self, factor)

    def to_units(self):
        return True

    def __str__(self):
        return 'DriveProlongationTransaction'


class DEx:
    def __init__(self):
        self.xpx = 1000000
        self.units = 1000000
        self.fee = 0.00
        self.courses = []

    def get_liquidity(self):
        return self.xpx * self.units

    def fill_transaction(self, tx: Transaction):
        inflation_params = {'a': 0.0, 'b': 0.0}
        transaction_type = type(tx)
        if transaction_type == ReplicatorOnBoardingTransaction:
            pass
        elif transaction_type == ReplicatorOffBoardingTransaction:
            pass
        elif transaction_type == DriveCreationTransaction:
            pass
        elif transaction_type == DriveProlongationTransaction:
            # print('Prolongation', tx.units)
            tx.units = tx.units * (1 - inflation_params['a'])
        elif transaction_type == PayToReplicatorsTransaction:
            # print('PayToRepl', tx.units)
            tx.units = tx.units / (1 - inflation_params['a']) * (1 + inflation_params['b'])
        elif transaction_type == DriveEndTransaction:
            pass
        if tx.to_units:
            tx.xpx = (self.get_liquidity() / (self.units - tx.units) - self.xpx)
            tx.fee = tx.xpx * self.fee / (1 - self.fee)
        else:
            tx.xpx = self.xpx - (self.get_liquidity() / (self.units + tx.units))
            tx.fee = tx.xpx * self.fee

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
        # self.xpx += tx.fee
        # self.units += tx.fee * course
        self.after_exchange_process(tx)
        self.courses.append(self.get_course())

class RatePredictor:
    def __init__(self, period, initial_course):
        self.low = -1
        self.high = 1
        self.period = period
        self.random_state = np.random.RandomState(64)
        self.rates = deque([(0, initial_course)])
        self.__add_next_rate(period)

    def __add_next_rate(self, time):
        rate = self.__generate_next_rate()
        self.rates.append((time, rate))
        if len(self.rates) > 2:
            self.rates.popleft()

    def __generate_next_rate(self):
        r = self.random_state.rand()
        r_mean = 0.5
        mean = (self.low + self.high) / 2
        r -= (r_mean - mean)
        amplitude = (self.high - self.low)
        # print('mean {}, amplitude {}'.format(mean, amplitude))
        r *= amplitude
        return r

    def get_rate_at(self, time):
        return -1
        while self.rates[-1][0] < time:
            self.__add_next_rate(self.rates[-1][0] + self.period)
        x_0, y_0 = self.rates[0]
        x_1, y_1 = self.rates[-1]
        y = (time - x_0) * (y_1 - y_0) / (x_1 - x_0) + y_0
        return y

class FlowGenerator:
    LINEAR_TYPE = 'Linear'
    LOG_TYPE = 'Log'
    SIGMOID_TYPE = 'Sigmoid'
    LINEAR_WITH_NEGATIVE_TYPE = 'LinearWithNegative'
    CONSTANT_TYPE = 'Constant'
    def __init__(self, type, rate_predictor: RatePredictor):
        self.rate_predictor = rate_predictor
        self.type = type

    def get_replicator_flow_proba(self, rate, time, standard_flow):
        real_rate = self.rate_predictor.get_rate_at(time)
        x = rate - real_rate
        if self.type == FlowGenerator.SIGMOID_TYPE:
            proba = 2 * 0.75 * 3 ** x / (1 + 0.75 * (3 ** x - 1)) - 1
            # print(time, self.get_max_flow(standard_flow) * proba)
            return proba
        elif self.type == FlowGenerator.LINEAR_TYPE:
            x = rate - real_rate
            flow = max(10 * (2 ** x - 1) + 1, 0)
            proba = flow * standard_flow / self.get_max_flow(standard_flow)
            return proba
        elif self.type == FlowGenerator.LINEAR_WITH_NEGATIVE_TYPE:
            x = rate - real_rate
            flow = 10 * (2 ** x - 1) + 1
            # print(time, flow)
            proba = flow * standard_flow / self.get_max_flow(standard_flow)
            return proba
        elif self.type == FlowGenerator.CONSTANT_TYPE:
            return 1
        else:
            flow = 1 + x
            proba = flow * standard_flow / self.get_max_flow(standard_flow)
            # print(time, proba * self.get_max_flow(standard_flow))
            return proba

    def get_client_flow_proba(self, rate, time, standard_flow):
        real_rate = self.rate_predictor.get_rate_at(time)
        x = rate - real_rate
        if self.type == FlowGenerator.SIGMOID_TYPE:
            return 2 * 0.75 * 3 ** (-x) / (1 + 0.75 * (3 ** (-x) - 1)) - 1
        elif self.type == FlowGenerator.LINEAR_TYPE:
            x = rate - real_rate
            flow = max(10 * (2 ** (-x) - 1) + 1, 0)
            # print('flow', flow)
            return flow * standard_flow / self.get_max_flow(standard_flow)
        elif self.type == FlowGenerator.LINEAR_WITH_NEGATIVE_TYPE:
            x = rate - real_rate
            flow = 10 * (2 ** (-x) - 1) + 1
            return flow * standard_flow / self.get_max_flow(standard_flow)
        elif self.type == FlowGenerator.CONSTANT_TYPE:
            return 1
        else:
            flow = 1 - x
            return flow * standard_flow / self.get_max_flow(standard_flow)

    def get_max_flow(self, standard_flow):
        #course diff log <= 3
        if self.type == FlowGenerator.SIGMOID_TYPE:
            return 2 * standard_flow
        elif self.type == FlowGenerator.LINEAR_TYPE:
            return 19 * standard_flow # if no inflation it is enough to set 19, else 20
        elif self.type == FlowGenerator.LINEAR_WITH_NEGATIVE_TYPE:
            return 19 * standard_flow
        elif self.type == FlowGenerator.CONSTANT_TYPE:
            return standard_flow
        else:
            return 4 * standard_flow


class RevenueRatePredictor:
    ADAPTIVE_TYPE = 'Adaptive Revenue Rate'
    CONSTANT_TYPE = 'No Revenue Rate'
    TEST_TYPE = 'Test'
    def __init__(self, type, rate_predictor: RatePredictor):
        self.rate_predictor = rate_predictor
        self.type = type

    def get_revenue_rate_at(self, time, exchange_rate):
        if self.type == RevenueRatePredictor.CONSTANT_TYPE:
            return 1
        else:
            # return 1
            # print(2 ** rate_predictor.get_rate_at(time) / exchange_rate)
            return 2 ** rate_predictor.get_rate_at(time) / exchange_rate

    def __str__(self):
        if self.type == RevenueRatePredictor.CONSTANT_TYPE:
            return 'No Revenue Rate'
        else:
            # return 1
            # print(2 ** rate_predictor.get_rate_at(time) / exchange_rate)
            return 'Adaptive Revenue Rate'


def perform_action(probability):
    return np.random.choice([True, False], p=[probability, 1 - probability])


class TransactionsGenerator:
    def __init__(self, init_course, rate_predictor: RatePredictor, revenue_type):
        self.init_course = init_course
        self.course = init_course
        self.flow_generator = FlowGenerator(FlowGenerator.LINEAR_TYPE, rate_predictor)
        self.revenue_rate_predictor = RevenueRatePredictor(revenue_type, rate_predictor)

    def notify_course_changed(self, course):
        self.course = course

    def get_log_course_change(self):
        return np.log2(self.course / self.init_course)

    def generate_transactions(self, billing_periods, standard_new_clients_per_billing_period):
        standard_new_replicators_per_billing_period = standard_new_clients_per_billing_period
        queue = []
        active_drives = 5000
        active_replicators = DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE * active_drives
        # for i in range(active_drives):
        #     heapq.heappush(queue, (np.random.rand(), PayToReplicatorsTransaction(active_replicators / active_drives, self.revenue_rate_predictor.get_revenue_rate_at(0, self.course))))
        heapq.heappush(queue,
                       (np.random.exponential(1
                                              / active_replicators * standard_new_replicators_per_billing_period), ReplicatorBoardingTransaction()))
        heapq.heappush(queue,
                       (np.random.exponential(1
                                              / active_replicators * standard_new_clients_per_billing_period), DriveCreationTransaction()))
        with open('{}.txt'.format(str(self.revenue_rate_predictor)), 'w') as f:
            while True:
                time, transaction = heapq.heappop(queue)
                if time > billing_periods:
                    break
                max_replicator_flow = self.flow_generator.get_max_flow(active_drives * DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE * standard_new_replicators_per_billing_period)
                max_client_flow = self.flow_generator.get_max_flow(
                    active_drives * standard_new_clients_per_billing_period)
                replicator_flow_proba = self.flow_generator.get_replicator_flow_proba(self.get_log_course_change(), time,
                                                                                      active_replicators * standard_new_replicators_per_billing_period)
                client_flow_proba \
                    = self.flow_generator.get_client_flow_proba(self.get_log_course_change(), time,
                                                                     active_drives * standard_new_clients_per_billing_period)
                # print(time, replicator_flow_proba, client_flow_proba)
                f.writelines(['{} {} {} {}\n'.format(time, max_client_flow, max_client_flow * client_flow_proba, self.course)])
                transaction_type = type(transaction)
                revenue_rate = self.revenue_rate_predictor.get_revenue_rate_at(time, self.course)
                if transaction_type == ReplicatorBoardingTransaction:
                    heapq.heappush(queue, (time + np.random.exponential(1
                                                                     / max_replicator_flow), ReplicatorBoardingTransaction()))
                    make_transaction = perform_action(np.abs(replicator_flow_proba))
                    #print(time, replicator_flow_proba)
                    if make_transaction:
                        if replicator_flow_proba > 0:
                            active_replicators += 1
                            yield time, ReplicatorOnBoardingTransaction()
                        elif active_replicators > 0:
                            active_replicators -= 1
                            yield time, ReplicatorOffBoardingTransaction()
                elif transaction_type == DriveCreationTransaction:
                    heapq.heappush(queue, (time + np.random.exponential(1
                                                                        / max_client_flow),
                                           DriveCreationTransaction()))
                    if client_flow_proba > 0:
                        # print(client_flow_proba)
                        make_transaction = perform_action(client_flow_proba)
                        if make_transaction:
                            active_drives += 1
                            transaction.specify_params(revenue_rate=revenue_rate)
                            replicators = min(active_replicators / (active_drives + 1),
                                             DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)
                            heapq.heappush(queue, (time + 1,
                                                   PayToReplicatorsTransaction(replicators, revenue_rate)))
                            yield time, transaction
                elif transaction_type == PayToReplicatorsTransaction:
                    replicators = min(active_replicators / active_drives,
                                    DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)
                    transaction.specify_params(replicators)
                    prolong = client_flow_proba >= 0
                    if client_flow_proba < 0:
                        proba = -client_flow_proba
                        prolong = perform_action(proba)
                    if prolong:
                        alpha = transaction.get_replicators() \
                                / DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE
                        beta = DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE * \
                            (revenue_rate + (alpha - 1) * transaction.revenue_rate)
                        heapq.heappush(queue, (time, DriveProlongationTransaction(
                            max(beta, 0))))
                    else:
                        'DriveEnd'
                        active_drives -= 1
                        heapq.heappush(queue, (time, DriveEndTransaction(
                            DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE -
                            transaction.get_replicators())))
                    yield time, transaction
                elif transaction_type == DriveProlongationTransaction:
                    replicators = min(active_replicators / active_drives,
                                     DriveCreationTransaction.MAX_REPLICATORS_PER_DRIVE)
                    heapq.heappush(queue, (time + 1,
                                           PayToReplicatorsTransaction(replicators, revenue_rate)))
                #     # print(transaction.units, PayToReplicatorsTransaction(replicators, revenue_rate).units)
                    yield time, transaction
                elif transaction_type == ReplicatorOffBoardingTransaction:
                    yield time, transaction
                elif transaction_type == DriveEndTransaction:
                    yield time, transaction


def add_transactions_to_statistics(tx, statistics):
    tx_name = str(tx)
    if tx_name not in statistics:
        statistics[tx_name] = 0
    statistics[tx_name] += 1


np.random.seed(16)
revenue_rates = [RevenueRatePredictor.ADAPTIVE_TYPE, RevenueRatePredictor.CONSTANT_TYPE]
# revenue_rates = [RevenueRatePredictor.CONSTANT_TYPE, RevenueRatePredictor.ADAPTIVE_TYPE]
# revenue_rates = [RevenueRatePredictor.ADAPTIVE_TYPE, RevenueRatePredictor.CONSTANT_TYPE]
# labels = ['No Revenue Rate']
for revenue_rate in revenue_rates:
    exchange = DEx()
    rate_predictor = RatePredictor(10, np.log2(exchange.get_course()))
    transactions = dict()
    times = []
    real_rates = []
    transactions_generator = TransactionsGenerator(exchange.get_course(), rate_predictor, revenue_rate)
    for time, tx in transactions_generator.generate_transactions(100, 0.01):
        times.append(time)

        real_rates.append(rate_predictor.get_rate_at(time))
        # if type(tx) == DriveCreationTransaction:
        #     print(time, end=' ')
        exchange.exchange(tx)
        add_transactions_to_statistics(tx, transactions)
        transactions_generator.notify_course_changed(exchange.get_course())
    plt.plot(times, exchange.courses, label=str(revenue_rate))

plt.plot(times, 2.0 ** np.array(real_rates), label='Desired Exchange Rate')
plt.legend()
plt.show()