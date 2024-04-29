## Задача 2. Обнаружение последовательности формальных подстановок при решении систем уравнений
**Вычисляемая последовательность формальных подстановок** - каждый элемент последовательности (уравнение) содержит в правой части только уже вычисленные неизвестные.
Введём матрицу $S_{n \times n}$ для правой части. Если последовательность вычисляема, то матрица $S$ - нижнетреугольная с нулями на диагонали.

Формальные подстановки, конечно, можно рассматривать как уравнения, но это увеличивает размер решаемой системы и может вызвать трудности для метода Ньютона (медленная сходимость, плохие начальные условия).
Целью является обнаружение формальных подстановок среди уравнений и их упорядочивание согласно определению формальных подстановок.
### Пример 0. Необходимое условие вычислимой последовательности 

$$ \left\{\begin{array}{lll}
x_1 & =A_1\\
x_2 & = \cos{x_1} + A_2\\
x_3 & =x_1^2 + x_1 + x_2\\
x_4 & =A_3\\
\end{array}\right. $$

Соответствующая матрица $S$:

$$ S = \begin{pmatrix}
   0 & 0 & 0 & 0 \\
   1 & 0 & 0 & 0 \\
   1 & 1 & 0 & 0 \\
   0 & 0 & 0 & 0 \\
\end{pmatrix} $$

Информацию о подстановках можно выразить в виде уравнения $x = Sx + C$. Выполнив процесс Гаусса над матрицей $S$ и решив уравнение $Sy = 0$, мы установим являются ли подстановки вычислимой последовательностью.

<h3 align="center">Алгоритм</h3>
  
1. Свести каждое уравнение системы к виду $x_i = f(x_1, \dots, x_n, t)$. $x_i$ не должен входить в правую часть выражения. Иначе, отнести это выражение в группу 'уравнений'.
2. Найти компоненты сильной связности с помощью алгоритма Тарьяна.
3. Удалить алгебраические петли. Запустить заново алгоритм Тарьяна.
4. Компоненты размера $1 \times 1$ являются подстановками, остальные - уравнениями.
5. Упорядочить последовательность и получить формальную вычислимую последовательность.
6. 
### Пример 1. Применение алгоритма
$$
\left\{\begin{array}{l}
x_{1}=-7x_{7}+x_{8}-2 \\
x_{2}=3x_{4} - 2x_{6}+1 \\
x_{3}=\frac{x_{5}}{2}+8 \\
x_{4}=\frac{x_{6}}{x_{1}}+12 \\
x_{5}=(x_{7}+x_{8}+x_{3})^2-23 \\
x_{6}=x_{3}+x_{2}-3 \\
x_{7}=2 x_{3}+2 \\
x_{8}=(x_{7}-2)^3+2
\end{array}\right.
$$
#### Визуализация графа

![alt text](imgs/graph3.png)

#### Об удалении петель
В примере 1 можно выделить две алгебраические петли:

$$ 1. \ x_3 \Longrightarrow x_5 \Longrightarrow x_7 \Longrightarrow x_8 \Longrightarrow x_3 $$

$$ 2. \ x_2 \Longrightarrow x_4 \Longrightarrow x_6 \Longrightarrow x_2 $$

Определение **циклового порядка** ($CO$, cycling order) := max {количество входящих ребер из вершин компоненты сильной связности (КСС); количество исходящих ребер в вершины текущей КСС}.

Необходимо выбрать вершину с максимальный цикловым порядком, отнести её в разряд 'уравнений' и **удалить** из графа. $CO(x_5) = 3$, равный максимальному, её и удалим. 
Если вершин с максимальным цикловым порядком несколько, то выберем ту, что первая встретилась в алгоритме Тарьяна. В данном случае, удалим вершину $x_2$ с порядком 2, которая встретилась первой. \\

#### Код
```python
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

tsk = SubstitutionDetector(dim=8)
tsk.set_matrix(M)
print(tsk.scc)
tsk.break_loops()
tsk.get_answer()
```
Вывод программы:
```
[[6, 7, 4, 2], [0], [1, 5, 3]] # компоненты сильной связности (КСС)
Для цикла [2, 4, 7, 6] будет удалена вершина: 4. Причина: Максимальный цикловый порядок
Для цикла [1, 3, 5] будет удалена вершина: 1. Причина: Максимальный цикловый порядок, первым встретился в алгоритме Тарьяна
Индексы уравнений: [4, 1]
Индексы(!) подстановок: [2, 6, 7, 0, 5, 3]
Индексы(!) подстановок, которые действительно есть в системе: []
```
#### Результат работы
Уравнения:

$$
\left\{\begin{array}{l}
x_{2}=3x_{4} - 2x_{6}+1 \\
x_{5}=(x_{7}+x_{8}+x_{3})^2-23 \\
\end{array}\right.
$$

Подстановки:

$$
\left\{\begin{array}{l}
x_{3}=\frac{x_{5}}{2}+8 \\
x_{7}=2 x_{3}+2 \\
x_{8}=(x_{7}-2)^3+2 \\
x_{1}=-7x_{7}+x_{8}-2 \\
x_{6}=x_{3}+x_{2}-3 \\
x_{4}=\frac{x_{6}}{x_{1}}+12 \\
\end{array}\right.
$$

**Обратите внимание на порядок подстановок: теперь условие вычислимости соблюдается.** 
