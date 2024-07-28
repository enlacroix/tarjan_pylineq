import networkx as nx
import numpy as np
from tarjan.utils import tarjan, matr2dct


class SubstitutionDetector:
    def __init__(self, dim):
        self.translator = dict()
        self.inverted_translator = dict()
        self.graph = None
        self.scc = None
        self.matrix = np.zeros((dim, dim))
        self.equations_indices = []
        self.substitution_indices = []
        self.in_system = []

    def add_equation(self, varnum: int, dependnums: list[int]):
        for i in dependnums:
            self.matrix[varnum - 1][i - 1] = 1
            self.in_system.append(varnum - 1)

    def set_matrix(self, matr):
        self.matrix = matr
        for i in range(self.matrix.shape[0]):
            if any(elem != 0 for elem in self.matrix[i]):
                self.in_system.append(i)
        self.initialize()

    def cycling_order(self, vertex_idx):
        """
        max(исходящие ребра, выходящие ребра) в вершину той же компоненты сильной связности
        :param vertex_idx:
        :return:
        """
        group = self.get_component_by_vertex(vertex_idx)
        ones_in_row = 0
        ones_in_column = 0

        for i, r in enumerate(self.matrix[vertex_idx, :]):
            if r == 1 and i in group:
                ones_in_row += 1

        for i, r in enumerate(self.matrix[:, vertex_idx]):
            if r == 1 and i in group:
                ones_in_column += 1

        return max(ones_in_row, ones_in_column)

    def _strong_connected_components(self):
        return tarjan(matr2dct(self.matrix))

    def initialize(self):
        self.scc = self._strong_connected_components()
        self.graph = nx.DiGraph(self.matrix)

    def get_cycles(self):
        result = []
        nodes = set()
        # определение: A simple cycle, or elementary circuit, is a closed path where no node appears twice.
        lst = list(nx.simple_cycles(self.graph))
        lst.sort(key=len, reverse=True)
        for cycle in lst:
            if all((n not in nodes for n in cycle)):
                result.append(cycle)
                for n in cycle:
                    nodes.add(n)
        return result

    def get_component_by_vertex(self, vertex_idx):
        cycles = self.get_cycles()
        for cycle in cycles:
            if vertex_idx in cycle:
                return cycle

    def find_to_delete(self, cycle):
        dct = dict(zip(cycle, [self.cycling_order(c) for c in cycle]))
        inv_dct = {}
        for k, v in dct.items():
            inv_dct[v] = [key for key in dct if dct[key] == v]
        # print(dct)
        # print(inv_dct)
        maxKey = max(inv_dct.keys())
        candidates = inv_dct[maxKey]
        if len(candidates) == 1:
            return candidates[0], False
        group = self.get_component_by_vertex(candidates[0])
        return min(candidates, key=group.index), True

    def break_loops(self):
        N = self.matrix.shape[0]
        self.translator = dict(zip(list(range(N)), list(range(N))))
        for cycle in self.get_cycles():
            num, reason = self.find_to_delete(cycle)
            self.equations_indices.append(num)
            print(f'Для цикла {cycle} будет удалена вершина: {num}. Причина: Максимальный цикловый порядок' + ', первым встретился в алгоритме Тарьяна' * reason)
            self.matrix = np.delete(self.matrix, num, axis=0)
            self.matrix = np.delete(self.matrix, num, axis=1)
            del self.translator[num]
            for k in self.translator:
                if k > num:
                    self.translator[k] -= 1
        self.inverted_translator = {v: k for k, v in self.translator.items()}
        self.initialize()

    def get_answer(self):
        print(f'Индексы уравнений: {self.equations_indices}')
        for sublist in self.scc:
            if len(sublist) > 1:
                print(f'{sublist} не подстановка, а видимо уравнение. Добавить в eq_indices?')
                continue
            self.substitution_indices.append(self.inverted_translator[sublist[0]])
        print(f'Индексы(!) подстановок: {self.substitution_indices}')
        print(f'Индексы(!) подстановок, которые действительно есть в системе: {[i for i in self.substitution_indices if i in self.in_system]}')


def test_1():
    tsk = SubstitutionDetector(8)
    tsk.add_equation(1, [2, 5])
    tsk.add_equation(3, [1, 6])  # delete 1
    tsk.add_equation(2, [4, 7])
    tsk.add_equation(4, [3, 8])
    tsk.add_equation(6, [7])

    tsk.initialize()
    tsk.break_loops()
    tsk.get_answer()
    return tsk


def test_2():
    M = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],

    ])

    tsk = SubstitutionDetector(8)
    tsk.set_matrix(M)
    print(tsk.scc)
    tsk.break_loops()
    tsk.get_answer()


if __name__ == '__main__':
    test_2()

