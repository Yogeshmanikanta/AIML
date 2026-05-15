import numpy as np
import matplotlib.pyplot as plt


data=np.random.rand(1000)

mean=sum(data)/len(data)

c=mean
s=0
for i in  data:
    t=0
    t=i-c
    t=t*t
    s+=t
var=s/len(data)

std=var**0.5


print("manual mean:",mean)
print("numpy",np.mean(data))

print("manual var:",var)
print("numpy",np.var(data))

print("manual std:",std)
print("numpy",np.std(data))


#ploting the histogram of the data
plt.hist(data,bins=30)
plt.title("histogram of the data")
plt.show()
