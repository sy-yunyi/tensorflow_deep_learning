# class demo():
#     def add(self,num1,num2):
#         return num1+num2
#     def multiply(self,num1,num2):
#         return num1*num2
#
# d = demo()
# print(d.add(2,3))
# print(d.multiply(3,4))

num = [str(num) for num in range(34)]
k = str(3)
count = 0
for i in num:
    if k in i :
        count += i.count(k)
print (count)