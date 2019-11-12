import matplotlib.pyplot as plt
names=['iphone','p20','mate20','v20','mix3']
num_list = [0.8927,0.978,0.9862,0.995,0.8972]
a=plt.bar(range(len(num_list)), num_list,tick_label=names)

plt.xlabel("type")  #设置X轴Y轴名称
plt.ylabel("rate")

#使用text显示数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

autolabel(a)
plt.ylim(0.5,1.1)
plt.show()