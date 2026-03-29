import time

from matplotlib.pylab import sqrt
positions = []
times = []
while True:
        inp = input("Enter x,y or 'stop':")
        if inp.lower() == 'stop':
            break
        x_str, y_str =inp.split(",")
        x, y = int(x_str),int(y_str)
        positions.append((x,y))
        times.append(time.time())
dx = positions[-1][0] - positions[-2][0]
dy = positions[-1][1] - positions[-2][1]
displacement = sqrt(dx**2 + dy**2)
dt = times[-1] - times[-2]  
speed = displacement / dt
    