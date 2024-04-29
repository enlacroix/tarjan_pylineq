import itertools
import numpy as np
from numpy.typing import NDArray

from tarjan.utils import tarjan_algorithm_with_start, matr2dct, colorGraph


class MinimalSubsetDetector:
    def __init__(self, matrix: NDArray, omitOnes: bool = True):
        assert matrix.shape[0] == matrix.shape[1], 'Матрица должна быть квадратной'
        permuted = MinimalSubsetDetector.permuteMatrix(matrix)
        if permuted is None:
            raise ValueError('У переданной матрицы не существует такой перестановки строк, что на главной диагонали стоят единицы.')
        self.matrix = permuted
        self.size = self.matrix.shape[0]
        self.hasRoot = False
        if omitOnes:
            for i in range(self.size):
                self.matrix[i][i] = 0

    @staticmethod
    def permuteMatrix(matrix) -> NDArray | None:
        """
        Находит такую перестановку строк, что на главной диагонали должны стоять только 1.
        O(k^N), где k - примерное кол-во 1 в строке.
        :param matrix:
        :return:
        """
        result = []
        for row in matrix:
            indices = np.where(row == 1)[0]
            result.append(indices.tolist())
        result_dict = {}
        for i, sublist in enumerate(result):
            for element in sublist:
                if element not in result_dict:
                    result_dict[element] = [i]
                else:
                    result_dict[element].append(i)
        result_list = [result_dict.get(i, []) for i in range(len(result))]
        combinations = itertools.product(*result_list)
        valid_perm = next(comb for comb in combinations if len(set(comb)) == len(comb))
        if valid_perm is None:
            print('Такой перестановки не существует.')
            return
        print(f'Необходимая перестановка: {valid_perm}.')
        return matrix[list(valid_perm)]

    @classmethod
    def initfromTeX(cls, dimension: int, texcode: str, omitOnes: bool = True):
        """
        Пример:
        1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 \\
        1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 \\
        1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
        1 & 1 & 0 & 0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 1
        :param omitOnes: опустить единицы на главной диагонали матрицы (убрать петли).
        :param dimension: размерность матрицы
        :param texcode: код для матрицы
        :return:
        """
        tex_matrix = texcode.replace("\\", '&').replace("\n", "")
        elements = tex_matrix.split("&")
        int_elements = [int(element.strip()) for element in elements if len(element)]
        return cls(np.array(int_elements).reshape(dimension, dimension), omitOnes=omitOnes)

    def addRootNode(self, interested_nodes: list[int]):
        """
        Добавление корневой вершины R, которая связана ребрами с interested_nodes. Применяется, когда
        нужно узнать минимальный набор уравнений для нахождения группы неизвестных.
        :param interested_nodes:
        :return:
        """
        self.hasRoot = True
        row = [0] * self.size
        for n in interested_nodes:
            row[n] = 1
        new_row = np.array(row)
        extended_matrix = np.vstack([self.matrix, new_row])
        new_column = np.array([0] * (self.size + 1))
        new_column = new_column[:, np.newaxis]
        self.matrix = np.hstack((extended_matrix, new_column))

    def find(self, node: int = None):
        if node is None and not self.hasRoot:
            raise ValueError('Укажите индекс переменной, которую вы хотите найти.')
        return tarjan_algorithm_with_start(matr2dct(self.matrix), start_node=self.size if self.hasRoot else node)

    def color_SCC(self, scc):
        return colorGraph(self.matrix, scc)


def test1():
    detector = MinimalSubsetDetector.initfromTeX(
        8,
        r'1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\'
        r'1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 \\'
        r'0 & 0 & 1 & 0 & 0 & 1 & 0 & 0 \\'
        r'1 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\'
        r'0 & 1 & 0 & 0 & 1 & 0 & 0 & 0 \\'
        r'1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\'
        r'1 & 1 & 0 & 0 & 0 & 0 & 1 & 0 \\'
        r'0 & 0 & 1 & 0 & 0 & 0 & 0 & 1'
    )
    # Для поиска минимального набора уравнений для переменных у1 и у2.
    # detector.addRootNode([6, 7])
    answers = detector.find(6)  # найти для у1
    print(answers)
    detector.color_SCC(answers)


if __name__ == '__main__':
    test1()
