import sys
import os

print('当前 Python 解释器路径：')
print(sys.executable)
r"""
当前 Python 解释器路径：
C:\Users\jpch89\AppData\Local\Programs\Python\Python36\python.EXE
"""

print()
print('当前 Python 解释器目录：')
print(os.path.dirname(sys.executable))
