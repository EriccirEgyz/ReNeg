import os

save_dir = 'responses'
dirs = os.listdir(save_dir)

num = 0
all_num = 532761
for d in dirs:
    if d.endswith('.txt') and d.startswith('_id_'):
        lines = open(f'{save_dir}/{d}').readlines()
        num += len(lines)

print(num)
print('Progress Ratio: {:.2f}%'.format(num / all_num * 100))