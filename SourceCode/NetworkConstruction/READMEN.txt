                程序运行的顺序和逻辑进行解释和说明
1. 现有构建网络程序
+ (1) LayerDataProcess.py (2) Network.py (3) RelationConstruct.py (4) NetworkConstruct.py (5) RollingConstruction_multiprocess.py (6) RollingConstruction_main.py

2. 运行说明
(1) LayerDataProcess.py : 对上市公司中ST，非ST数据进行处理和提取，并且需要有对应的时间
(2) Network.py : 一个我们始终使用且不变的 Class 对象
(3) RelationConstruct.py : 包含两个部分，一个部分是关系提取，一个部分是关系构建（调用关系提取，形成一个复杂关系网）
(4) NetworkConstruct.py : 根据关系数据，形成网络（调用Network）
(5) RollingConstruction_multiprocess.py : 多线程构建不同时期的网络
(6) RollingConstruction_main : 单线程



3. 运行逻辑
RollingConstruction_multiprocess ——> 调用 ——> NetworkConstruct ——> 调用 ——> RelationConstruct ——> 通过DCC_garch模型进行计算
               |                                  |
              调用                                调用
               |                                  |
        LayerDataProcess                        Network