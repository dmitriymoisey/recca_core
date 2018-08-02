import os


def check_task_name(name):
    if not (os.path.exists(f'./user/task_db/{name}')):
        print(f'Directory : \'./user/task_db/{name}\' DOESN\'t  EXISTS!')
        return False
    else:
        return True
