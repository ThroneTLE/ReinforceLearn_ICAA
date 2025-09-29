import numpy as np
import random

# 定义一个名为 GridWorld_v1 的类，用于创建和管理一个网格世界环境
class GridWorld_v1(object):
    # 类的注释：
    # 这是一个初版的网格世界（GridWorld）环境。
    # 它的特点是“确定性”的，即在某个状态执行一个动作，其结果是唯一确定的，没有随机性。
    # 这个环境被设计用来计算和学习基础的强化学习算法，如策略迭代（Policy Iteration）和价值迭代（Value Iteration）。
    #
    # 环境构成：
    # - n行, m列的网格
    # - 随机生成若干个禁止区域（Forbidden Area）和目标区域（Target）
    #
    # 动作空间（Action Space）:
    # A1 (动作0): 向上移动
    # A2 (动作1): 向右移动
    # A3 (动作2): 向下移动
    # A4 (动作3): 向左移动
    # A5 (动作4): 保持原地

    # --- 类的核心属性 ---

    # stateMap: 一个二维列表，用于将网格的(行, 列)坐标映射到一个唯一的状态(State)编号。
    # 例如，在一个4x5的网格中，(0,0)是状态0，(0,1)是状态1，(3,4)是状态19。
    stateMap = None

    # scoreMap: 一个二维numpy数组，存储进入每个网格单元能获得的即时奖励(Reward)。
    # 它的维度与网格大小相同。例如，普通格子的奖励为0，目标格子为1，禁止区域为-10。
    scoreMap = None

    # score: 到达目标区域时获得的奖励值。
    score = 0

    # forbiddenAreaScore: 进入禁止区域时受到的惩罚值（负奖励）。
    forbiddenAreaScore = 0


    def __init__(self,rows=4, columns=5, forbiddenAreaNums=3, targetNums=1, seed = -1, score = 1, forbiddenAreaScore = -1, desc=None):
        """
        构造函数，用于初始化一个网格世界实例。
        它支持两种创建模式：从描述（desc）创建固定世界，或随机创建一个世界。

        参数:
        rows (int): 网格的行数。
        columns (int): 网格的列数。
        forbiddenAreaNums (int): 随机生成的禁止区域数量。
        targetNums (int): 随机生成的目标区域数量。
        seed (int): 随机数种子。用于保证随机生成的世界是可复现的。如果seed值相同，生成的地图也相同。
        score (int/float): 到达目标区域的奖励值。
        forbiddenAreaScore (int/float): 进入禁止区域的惩罚值。
        desc (list of lists of str): 一个用于描述固定地图的二维列表。例如 [['', 'T'], ['#', '']]
                                      其中 'T' 代表目标，'#' 代表禁止区域。如果提供了此参数，则忽略前面的随机生成参数。
        """
        # 1. 初始化奖励值
        self.score = score
        self.forbiddenAreaScore = forbiddenAreaScore

        # 2. 判断创建模式：是根据描述创建还是随机创建
        if(desc != None):
            # --- 模式一：根据提供的`desc`参数创建固定的网格世界 ---
            self.rows = len(desc)
            self.columns = len(desc[0])
            l = []
            # 遍历desc，将字符转换为对应的奖励分数，并构建scoreMap
            for i in range(self.rows):
                tmp = []
                for j in range(self.columns):
                    # 根据desc中的字符设置奖励值
                    char = desc[i][j]
                    if char == '#':
                        tmp.append(forbiddenAreaScore)
                    elif char == 'T':
                        tmp.append(score)
                    else:
                        tmp.append(0)
                l.append(tmp)
            self.scoreMap = np.array(l)
            # 创建stateMap，将(i, j)坐标映射到状态编号 i * columns + j
            self.stateMap = [[i*self.columns+j for j in range(self.columns)] for i in range(self.rows)]
            return # 创建完成，直接返回

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

        # 初始化一个一维列表g，用于临时存放每个格子的奖励值
        self.g = [0 for i in range(self.rows * self.columns)]

        # 根据打乱后的列表l，设置禁止区域
        for i in range(forbiddenAreaNums):
            self.g[l[i]] = forbiddenAreaScore

        # 接着设置目标区域
        for i in range(targetNums):
            self.g[l[forbiddenAreaNums+i]] = score

        # 将一维的奖励列表g转换为(rows, columns)形状的二维numpy数组，作为scoreMap
        self.scoreMap = np.array(self.g).reshape(rows,columns)

        # 创建stateMap
        self.stateMap = [[i*self.columns+j for j in range(self.columns)] for i in range(self.rows)]

    def show(self):
        """
        可视化函数，将当前的网格世界布局打印到控制台。
        """
        print("Grid World Layout:")
        for i in range(self.rows):
            s = "" # 用于拼接每一行的字符串
            for j in range(self.columns):
                # 定义一个字典，将奖励分数映射到对应的emoji表情
                tmp = {0:"⬜️", self.forbiddenAreaScore:"🚫", self.score:"✅"}
                # 根据当前格子的奖励值，选择对应的表情并拼接到字符串s
                s = s + tmp[self.scoreMap[i][j]]
            print(s) # 打印一整行

    def getScore(self, nowState, action):
        """
        环境的核心函数，模拟智能体的状态转移过程。
        根据当前状态和执行的动作，返回获得的奖励和下一个状态。

        参数:
        nowState (int): 智能体当前所在位置的状态编号。
        action (int): 智能体执行的动作编号 (0:上, 1:右, 2:下, 3:左, 4:不动)。

        返回:
        tuple: 一个包含(奖励, 下一个状态)的元组。
        """
        # 1. 将一维的状态编号解码为二维的(x, y)坐标
        nowx = nowState // self.columns
        nowy = nowState % self.columns

        # 错误检查，确保当前状态坐标和动作在有效范围内
        if(nowx<0 or nowy<0 or nowx>=self.rows or nowy>=self.columns):
            print(f"coordinate error: ({nowx},{nowy})")
        if(action<0 or action>=5 ):
            print(f"action error: ({action})")

        # 2. 定义动作对坐标的影响
        # actionList的索引对应动作编号，值是(行变化, 列变化)
        # (上, 右, 下, 左, 不动)
        actionList = [(-1,0), (0,1), (1,0), (0,-1), (0,0)]

        # 3. 计算执行动作后理论上的新坐标
        tmpx = nowx + actionList[action][0]
        tmpy = nowy + actionList[action][1]

        # 4. 边界检查：判断新坐标是否超出了网格范围
        if(tmpx<0 or tmpy<0 or tmpx>=self.rows or tmpy>=self.columns):
            # 如果撞墙，则状态不发生改变，并返回一个-1的惩罚
            return -1, nowState

        # 5. 如果没有撞墙，返回新位置的奖励和新位置的状态编号
        # self.scoreMap[tmpx][tmpy] 是进入新格子的即时奖励
        # self.stateMap[tmpx][tmpy] 是新格子的状态编号
        return self.scoreMap[tmpx][tmpy], self.stateMap[tmpx][tmpy]

    def showPolicy(self, policy):
        """
        可视化函数，将传入的策略(policy)以emoji箭头的形式打印在网格上。

        参数:
        policy (list/array): 一个长度为总状态数的列表或数组。
                               policy[i]的值代表在状态i时，应该执行的动作编号。
        """
        print("Policy Visualization:")
        s = ""
        # 遍历所有状态
        for i in range(self.rows * self.columns):
            # 将状态编号转换为坐标
            nowx = i // self.columns
            nowy = i % self.columns

            # 判断当前格子的类型并选择合适的emoji
            if(self.scoreMap[nowx][nowy] == self.score):
                # 如果是目标格子，直接显示目标符号
                s = s + "✅"
            elif(self.scoreMap[nowx][nowy] == 0):
                # 如果是普通格子，使用普通箭头
                tmp = {0:"⬆️", 1:"➡️", 2:"⬇️", 3:"⬅️", 4:"🔄"}
                s = s + tmp[policy[i]]
            elif(self.scoreMap[nowx][nowy] == self.forbiddenAreaScore):
                # 如果是禁止区域，使用加粗的箭头，表示“紧急离开”
                tmp = {0:"⏫️", 1:"⏩️", 2:"⏬", 3:"⏪", 4:"🔄"}
                s = s + tmp[policy[i]]

            # 如果到达一行的末尾，则打印该行并重置字符串s
            if(nowy == self.columns-1):
                print(s)
                s = ""