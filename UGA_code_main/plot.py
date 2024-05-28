import re
import matplotlib.pyplot as plt


log_file_path = 'flir.out'


mean_ap_values = []
with open(log_file_path, 'r') as file:
    for line in file:
        match = re.search(r'Mean AP = (\d+\.\d+)', line)
        if match:
            mean_ap_values.append(float(match.group(1)))


if not mean_ap_values:
    print("No 'Mean AP' values found in the log file.")
else:
    # 绘制折线图
    plt.plot(mean_ap_values, marker='o')
    plt.title('Mean AP Values Over Time')
    plt.xlabel('Index')
    plt.ylabel('Mean AP')
    plt.grid(True)
    plt.savefig("1.jpg")
