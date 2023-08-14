'''
@Author: haozhi chen
@Date: 2021-12-29
@Target: implement your own exception

这里我们自己编写一个exception继承Exception即可
'''

class Myexception(Exception):

    def __init__(self,param):
        self.param = param

    def __str__(self):
        print("这里唤起了你可能遇到的错误，不用紧张，我们只对当前可能的错误进行了中断，具体信见："+str(self.param))

