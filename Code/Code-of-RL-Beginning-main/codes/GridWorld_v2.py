import random

import numpy as np


# 类的注释：
# 与v1版本的主要区别：
# 1. 支持随机性策略（Stochastic Policy）：
#    - v1的策略是确定性的，即在每个状态（state）只会选择一个固定动作。
#    - v2的策略是随机性的，即在同一个状态，可以按照一定的概率分布选择不同的动作。
#    - 因此，v2的策略矩阵（policy matrix）的维度是 (状态总数, 动作数)，例如 (25, 5)。
#      矩阵的每一行代表一个状态，该行所有元素的和为1，分别表示选择5个动作的概率。
#    - 在可视化策略时，为了直观，仍然只显示每个状态下概率最大的那个动作。
#
# 2. 引入轨迹（Trajectory）概念：
#    - 新增了 getTrajectoryScore 方法。
#    - 该方法可以根据一个给定的随机性策略，从一个初始状态开始，模拟（采样）出一条包含若干步的完整路径，并记录下每一步的详细信息。

class GridWorld_v2(object):
    # n行，m列，随机若干个forbiddenArea，随机若干个target
    # 动作空间 (Action Space):
    # A1 (动作0): 向上移动 (move upwards)
    # A2 (动作1): 向右移动 (move rightwards)
    # A3 (动作2): 向下移动 (move downwards)
    # A4 (动作3): 向左移动 (move leftwards)
    # A5 (动作4): 保持不变 (stay unchanged)

    # --- 类的核心属性 ---

    stateMap = None  # 二维列表，将网格的(行, 列)坐标映射到唯一的状态(State)编号。
    scoreMap = None  # 二维numpy数组，存储进入每个网格单元能获得的即时奖励(Reward)。
    score = 0  # 到达目标区域时获得的奖励值。
    forbiddenAreaScore = 0  # 进入禁止区域时受到的惩罚值（负奖励）。

    def __init__(self, rows=4, columns=5, forbiddenAreaNums=3, targetNums=1, seed=-1, score=1, forbiddenAreaScore=-1,
                 desc=None):
        """
        构造函数，用于初始化一个网格世界实例。
        (此函数与v1版本完全相同)

        参数:
        rows (int): 网格的行数。
        columns (int): 网格的列数。
        forbiddenAreaNums (int): 随机生成的禁止区域数量。
        targetNums (int): 随机生成的目标区域数量。
        seed (int): 随机数种子，用于保证随机生成的世界是可复现的。
        score (int/float): 到达目标区域的奖励值。
        forbiddenAreaScore (int/float): 进入禁止区域的惩罚值。
        desc (list of lists of str): 一个用于描述固定地图的二维列表，例如 [['', 'T'], ['#', '']]。
        """

        # 初始化奖励值
        self.score = score
        self.forbiddenAreaScore = forbiddenAreaScore

        # 判断创建模式：是根据描述创建还是随机创建
        if (desc != None):
            # --- 模式一：根据提供的`desc`参数创建固定的网格世界 ---
            self.rows = len(desc)
            self.columns = len(desc[0])
            l = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    char = desc[i][j]
                    if char == '#':
                        tmp.append(forbiddenAreaScore)
                    elif char == 'T':
                        tmp.append(score)
                    else:
                        tmp.append(0)
                l.append(tmp)
            self.scoreMap = np.array(l)
            self.stateMap = [[i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]
            return  # 创建完成，直接返回

        # --- 模式二：随机创建网格世界 ---
        self.rows = rows
        self.columns = columns
        self.forbiddenAreaNums = forbiddenAreaNums
        self.targetNums = targetNums
        self.seed = seed

        # 设置随机数种子以保证可复现性
        random.seed(self.seed)
        # 创建一个从0到(rows * columns - 1)的列表，代表所有格子的索引
        l = [i for i in range(self.rows * self.columns)]
        # 使用shuffle函数将列表随机打乱
        random.shuffle(l)

        self.g = [0 for i in range(self.rows * self.columns)]
        # 根据打乱后的列表l，设置禁止区域
        for i in range(forbiddenAreaNums):
            self.g[l[i]] = forbiddenAreaScore
        # 接着设置目标区域
        for i in range(targetNums):
            self.g[l[forbiddenAreaNums + i]] = score

        # 将一维的奖励列表g转换为(rows, columns)形状的二维numpy数组，作为scoreMap
        self.scoreMap = np.array(self.g).reshape(rows, columns)
        # 创建stateMap
        self.stateMap = [[i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]

    def show(self):
        """
        可视化函数，将当前的网格世界布局打印到控制台。
        (此函数与v1版本完全相同)
        """
        for i in range(self.rows):
            s = ""  # 用于拼接每一行的字符串
            for j in range(self.columns):
                # 定义一个字典，将奖励分数映射到对应的emoji表情
                tmp = {0: "⬜️", self.forbiddenAreaScore: "🚫", self.score: "✅"}
                # 根据当前格子的奖励值，选择对应的表情并拼接到字符串s
                s = s + tmp[self.scoreMap[i][j]]
            print(s)  # 打印一整行

    def getScore(self, nowState, action):
        """
        环境的核心函数，模拟智能体的状态转移过程。
        根据当前状态和执行的动作，返回获得的奖励和下一个状态。
        (此函数与v1版本完全相同)

        参数:
        nowState (int): 智能体当前所在位置的状态编号。
        action (int): 智能体执行的动作编号 (0:上, 1:右, 2:下, 3:左, 4:不动)。

        返回:
        tuple: 一个包含(奖励, 下一个状态)的元组。
        """
        # 将一维的状态编号解码为二维的(x, y)坐标
        nowx = nowState // self.columns
        nowy = nowState % self.columns

        # 错误检查
        if (nowx < 0 or nowy < 0 or nowx >= self.rows or nowy >= self.columns):
            print(f"coordinate error: ({nowx},{nowy})")
        if (action < 0 or action >= 5):
            print(f"action error: ({action})")

        # 定义动作对坐标的影响 (上, 右, 下, 左, 不动)
        actionList = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        # 计算执行动作后理论上的新坐标
        tmpx = nowx + actionList[action][0]
        tmpy = nowy + actionList[action][1]

        # 边界检查：判断新坐标是否超出了网格范围
        if (tmpx < 0 or tmpy < 0 or tmpx >= self.rows or tmpy >= self.columns):
            # 如果撞墙，则状态不发生改变，并返回一个-1的惩罚
            return -1, nowState
        # 如果没有撞墙，返回新位置的奖励和新位置的状态编号
        return self.scoreMap[tmpx][tmpy], self.stateMap[tmpx][tmpy]

    def getTrajectoryScore(self, nowState, action, policy, steps, stop_when_reach_target=False):
        """
        根据给定的策略，进行采样，生成一条轨迹(Trajectory)。

        参数:
        nowState (int): 轨迹的初始状态。
        action (int): 轨迹的初始动作。
        policy (numpy.ndarray): 一个 (状态数, 动作数) 的二维数组。
                                policy[s] 是一个包含5个概率值的列表，代表在状态s下选择每个动作的概率。
        steps (int): 要采样的步数。
        stop_when_reach_target (bool): 一个标志，如果为True，则在到达目标区域时提前终止采样。

        返回:
        list: 一个列表，其中每个元素都是一个元组 (nowState, nowAction, score, nextState, nextAction)，
              记录了轨迹中每一步的详细信息。
              注意：返回列表的长度是 steps+1，因为它包含了初始状态这一步。
        """

        res = []  # 用于存储轨迹结果的列表
        nextState = nowState  # 初始化下一个状态为当前状态
        nextAction = action  # 初始化下一个动作为当前动作

        # 如果设置了到达目标即停止，将步数设置为一个非常大的数，以确保能到达
        if stop_when_reach_target == True:
            steps = 20000

        # 循环采样指定的步数 (steps+1是因为要包含初始状态)
        for i in range(steps + 1):
            # 将上一轮的 "下一个状态/动作" 更新为当前轮的 "当前状态/动作"
            nowState = nextState
            nowAction = nextAction

            # 根据当前的状态和动作，从环境中获取即时奖励和转移到的下一个状态
            score, nextState = self.getScore(nowState, nowAction)

            # 核心：根据策略进行随机采样，决定在 "nextState" 要执行的动作 "nextAction"
            # np.random.choice 会根据 policy[nextState] 中定义的概率分布，随机选择一个动作
            # range(5) -> [0, 1, 2, 3, 4]，代表所有可能的动作索引
            # p=policy[nextState] -> 指定了从这5个动作中抽样的概率
            nextAction = np.random.choice(range(5), size=1, replace=False, p=policy[nextState])[0]

            # 将这一步采样的所有信息作为一个元组存入结果列表
            res.append((nowState, nowAction, score, nextState, nextAction))

            # 如果设置了提前终止，则进行检查
            if (stop_when_reach_target):
                # 获取当前状态的坐标
                nowx = nowState // self.columns
                nowy = nowState % self.columns
                # 检查当前状态是否是目标区域
                if self.scoreMap[nowx][nowy] == self.score:
                    return res  # 如果是，则直接返回已生成的轨迹

        return res  # 返回完整的轨迹

    def showPolicy(self, policy):
        """
        可视化函数，将传入的策略(policy)以emoji箭头的形式打印在网格上。
        由于策略是随机性的，这里只显示每个状态下概率最大的那个动作。

        参数:
        policy (numpy.ndarray): 一个(状态数, 动作数)的二维概率矩阵。
        """
        rows = self.rows
        columns = self.columns
        s = ""

        # 遍历所有状态
        for i in range(self.rows * self.columns):
            # 将状态编号转换为坐标
            nowx = i // columns
            nowy = i % columns

            # 判断当前格子的类型并选择合适的emoji
            if (self.scoreMap[nowx][nowy] == self.score):
                s = s + "✅"  # 目标区域

            elif (self.scoreMap[nowx][nowy] == 0):
                # 普通区域，使用普通箭头
                tmp = {0: "⬆️", 1: "➡️", 2: "⬇️", 3: "⬅️", 4: "🔄"}
                # np.argmax(policy[i]) 会找到在状态i下，概率最大的那个动作的索引
                s = s + tmp[np.argmax(policy[i])]

            elif (self.scoreMap[nowx][nowy] == self.forbiddenAreaScore):
                # 禁止区域，使用加粗的箭头
                tmp = {0: "⏫️", 1: "⏩️", 2: "⏬", 3: "⏪", 4: "🔄"}
                s = s + tmp[np.argmax(policy[i])]

            # 如果到达一行的末尾，则打印该行并重置字符串s
            if (nowy == columns - 1):
                print(s)
                s = ""