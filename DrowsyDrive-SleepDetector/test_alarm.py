import winsound
import time

print("Testing Windows alarm...")
print("You should hear the alarm now!")

for i in range(5):
    winsound.Beep(1000, 500)
    time.sleep(0.2)

print("Alarm test finished.")