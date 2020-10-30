import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Transaction:

    def __init__(self, on_board):
        self.units = 20
        self.type = type
        self.xpx = -1
        self.fee = 0
        self.on_board = on_board
        self.to_units = True


class DEx:
    def __init__(self):
        self.xpx = 100000
        self.units = 100000
        self.fee = 0.00
        self.courses = []
        self.to_units = 0
        self.to_xpx = 0
        self.on_board = 0
        self.on_drive = 0

    def get_liquidity(self):
        return self.xpx * self.units

    def fill_transaction(self, tx: Transaction):
        if tx.to_units:
            tx.xpx = (self.get_liquidity() / (self.units - tx.units) - self.xpx)
        else:
            units_after_fee = tx.units * (1 - self.fee)
            tx.xpx = self.xpx - (self.get_liquidity() / (self.units + units_after_fee))


    def get_course(self):
        return self.xpx / self.units

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
        if tx.to_units and tx.on_board:
            self.units += 4 * tx.units
            self.on_board += 1
        else:
            self.on_drive += 1
        self.courses.append(self.get_course())


def generate_transactions(n, open_close_ratio):
    if open_close_ratio < 1:
        open_close_ratio = 1
    print(open_close_ratio)
    opened = 0
    for i in range(n):
        if opened == 0:
            on_board = True
        else:
            on_board = np.random.choice([True, False], p=[open_close_ratio
                                                   / (open_close_ratio + 1), 1 / (open_close_ratio + 1)])
        if on_board:
            opened += 1
        else:
            opened -= 1
        yield Transaction(on_board)


np.random.seed(17)
exchanges = []
mean_courses = []
open_close_ratio = float(input('On-Boarding : Drive Orders Ratio'))
for i in range(1):
    print(i)
    try:
        exchange = DEx()
        for tx in generate_transactions(1000000, open_close_ratio):
            exchange.exchange(tx)
        plt.plot(exchange.courses)
        plt.show()
    except:
        pass
    # plt.plot(np.linspace(0, 1, len(exchange.courses)), exchange.courses)
    # plt.show()