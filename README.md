# peimg_generator

## 使用方法
- 将 peimg_generator.py 和所有需要处理的 radiant 测 PE 输出的 txt 文件置于同一目录
- 设置厚度、电极面积等关键参数 (若测试时正确输入可跳过)
- 设置绘图范围、图例等可选参数
- 运行 peimg_generator.py

你将得到：
1. 一系列 PE 曲线
2. Pmax, Pr 随电场变化的曲线
3. Wrec, η 随电场变化的曲线
4. Pmax, Pr, Wrec, η 随 E 变化曲线的数据 (可用于再次作图)

## 已实现功能
- 识别输入的数据是电场还是电压
- 识别 double bipolar 数据并选择其中一部分作图并计算储能密度
