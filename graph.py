import json
import matplotlib.pyplot as plt

#   og clip had 5780 frames over 5AM - 9PM => 16hrs
#   3:12 over 16hrs
#   57600 real secs compressed to 192 secs
#   300 realtime seconds / vid second
#   5 min realtime / vid second
#   cut 60 sec from start and 5 from end
#   300 mins realtime cut from start, 25 mins realtime cut from end
#   Trimmed time lapse: 10:00 AM - 8:35 PM
#   960 mins in 16 hrs
#   mins per frame = 960 / 5780 ~ .16
#   about 9.81 seconds per frame

def intToTime(num):
    numSecs = int(9.81 * num)
    numMins = int(numSecs / 60)
    hours = int(numMins / 60)
    mins = numMins % 60
    finalHour = hours + 10
    meridian = "am"
    if(finalHour > 12):
        finalHour = finalHour % 12
        if(finalHour == 0):
            finalHour = 12
        meridian = "pm"
    time = f'{finalHour}:{"{:02d}".format(mins)} {meridian}'
    return time

data = {}

for i in range(39):
    f = open(f'./data/framedataBatch{i}')
    update = json.load(f)
    data = {**data, **update}
    f.close

x = [int(k) for k in data.keys()]
y = [v for v in data.values()]

labels = [intToTime(i) for i in range(3831)]

plt.figure(figsize=(15, 10))  
plt.plot(x, y)  
plt.xticks(x[::100], labels[::100], rotation=45)

plt.title('Blue Sky Showing Through The Day')
plt.xlabel('Time')
plt.ylabel('Percentage')

plt.grid(True)
plt.show()