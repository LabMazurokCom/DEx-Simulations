import numpy as np
import matplotlib.pyplot as plt

class Transaction:

    made_transactions = 0

    def __init__(self):
        self.units = 20
        self.to_units = np.random.choice([True, False], 1, p=[0.5, 0.5])
        self.xpx = -1
        self.fee = 0


class DEx:
    def __init__(self, initial_course):
        self.units = 100000
        self.xpx = self.units * initial_course
        self.fee = 0.003
        self.courses = []
        self.to_units = 0
        self.to_xpx = 0

    def get_liquidity(self):
        return self.xpx * self.units

    def fill_transaction(self, tx: Transaction):
        if tx.to_units:
            tx.xpx = (self.get_liquidity() / (self.units - tx.units) - self.xpx)
            tx.fee = tx.xpx * self.fee / (1 - self.fee)
        else:
            tx.xpx = self.xpx - (self.get_liquidity() / (self.units + tx.units))
            tx.fee = tx.xpx * self.fee

    def exchange(self, tx:Transaction):
        self.fill_transaction(tx)
        if tx.to_units:
            self.xpx += tx.xpx
            self.units -= tx.units
        else:
            self.xpx -= tx.xpx
            self.units += tx.units
        course = self.units / self.xpx
        self.xpx += tx.fee
        self.units += tx.fee * course
        self.courses.append(self.xpx / self.units)

exchanges = []
mean_courses = []
initial_course = 2.94
for i in range(100):
    print(i)
    try:
        exchange = DEx(initial_course)
        for i in range(1000000):
            tx = Transaction ()
            exchange.exchange(tx)
        mean_courses.append(np.mean(np.log2(exchange.courses)))
        exchanges.append(exchange)
    except:
        pass
    # plt.plot(np.linspace(0, 1, len(exchange.courses)), exchange.courses)
    # plt.show()
plt.hist(np.array(mean_courses) - np.log2(initial_course))
plt.show()