x = 3
y = 3
z = 3

counter = 0

for i in range(z):
    print(str(counter) + " ----- ")
    for j in range(x):
        print("left x wall")
        print(str(j) + " " + str(0) + " " + str(i))
        print("right x wall")
        print(str(j) + " " + str(y - 1) + " " + str(i))
    for j in range(y):
        print("bottom y wall")
        print(str(0) + " " + str(j) + " " + str(i))
        print("up y wall")
        print(str(x - 1) + " " + str(j) + " " + str(i))
    counter += 1

# roof and floor caps
#for i in range(1, x - 1):
#    for j in range(1, y - 1):
#        obstacles.add((i, j, 0))
#        obstacles.add((i, j, z - 1))