import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Transaction:

    def __init__(self, on_board):
        self.units = 20
        self.xpx = -1
        self.fee = 0
        self.on_board = on_board
        self.to_units = True


class DEx:
    def __init__(self):
        self.xpx = 3000
        self.units = 3000
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
            if tx.on_board:
                print(tx.xpx / tx.units)
                print(self.xpx)
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


def generate_transactions(n):
    for i in range(n):
        yield Transaction(True)
    for i in range(n):
        yield Transaction(False)


exchanges = []
mean_courses = []
for i in range(1):
    print(i)
    try:
        exchange = DEx()
        for tx in generate_transactions(1000):
            exchange.exchange(tx)
        plt.plot(exchange.courses)
        plt.show()
    except Exception as e:
        print(e)
    # plt.plot(np.linspace(0, 1, len(exchange.courses)), exchange.courses)
    # plt.show()