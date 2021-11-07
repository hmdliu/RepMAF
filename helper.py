
import os
import sys
from shutil import move

def archive_log(dst_dir = './results/', log_dir = './'):
    file_list = os.listdir(log_dir)
    # print(file_list)
    for file in file_list:
        if file.endswith('.log'):
            exp = file.split('.')[0]
            grp = exp.split('_')[0]
            dst_path = os.path.join(dst_dir, grp, exp)
            log_path = os.path.join(log_dir, exp + '.log')
            with open(log_path, 'r') as f:
                res = f.read()[-300:].split('\n')
                end_flag = res[-3].endswith('mins')
            if sys.argv[2] == 'move':
                if os.path.isfile(log_path) and os.path.isdir(dst_path) and end_flag:
                    move(log_path, dst_path)
                    print('Exp [%s] archived.' % exp)
                else:
                    print('Exp [%s] skipped.' % exp)
            else:
                if end_flag:
                    print(dst_path)
                    print(log_path)
                    print(end_flag)

def dump_results(log_dir='./'):
    exp_list, acc_list, time_list = [], [], []
    for file in sorted(os.listdir(log_dir)):
        if not file.endswith('.log'):
            continue
        exp_id = file[:file.find('.log')]
        exp_list.append(exp_id)
        with open(file, 'r') as f:
            res = f.read()[-300:].split('\n')
            if not res[-3].endswith('mins'):
                print('Skipped [%s] since it\'s incomplete.' % exp_id)
                acc_list.append('TBD')
                time_list.append('TBD')
                continue
            # print(res)
            acc_list.append(res[-2][res[-2].find(':')+2:])
            time_list.append(res[-3][res[-3].find(':')+2:])
    
    print('\n[Summary]:')
    for i in range(len(exp_list)):
        print(exp_list[i], acc_list[i], time_list[i], sep='\t')
    print('\n[Exp_id]:\n%s' % ('\n'.join(exp_list)))
    print('\n[Accuracy]:\n%s' % ('\n'.join(acc_list)))
    print('\n[Time]:\n%s' % ('\n'.join(time_list)))

if __name__ == '__main__':
    # print(sys.argv)
    if sys.argv[1] == 'log':
        archive_log()
    elif sys.argv[1] == 'dump':
        dump_results()