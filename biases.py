import numpy as np
import matplotlib.pyplot as plt

# Load bias log
bias_log = np.genfromtxt("imu_biases.csv", delimiter=",", names=True)

t = bias_log["timestamp"]
bgx, bgy, bgz = bias_log["bg_x"], bias_log["bg_y"], bias_log["bg_z"]
bax, bay, baz = bias_log["ba_x"], bias_log["ba_y"], bias_log["ba_z"]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, bgx, label="bg_x")
plt.plot(t, bgy, label="bg_y")
plt.plot(t, bgz, label="bg_z")
plt.ylabel("gyro bias [rad/s]")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, bax, label="ba_x")
plt.plot(t, bay, label="ba_y")
plt.plot(t, baz, label="ba_z")
plt.ylabel("accel bias [m/s^2]")
plt.xlabel("time [s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
