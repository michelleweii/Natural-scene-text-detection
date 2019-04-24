import os

def checkModel(MP):
    if os.path.exists(MP):
        model_path = os.listdir(MP)
        if model_path is False:
            return False, 'NO MODEL'
        else:
            model_path.sort(key=lambda fn: os.path.getmtime(MP + '/' + fn),
                            reverse=True)
            flag = 0
            for m in model_path:
                if m.endswith('hdf5'):
                    Mname = os.path.join(MP, m)
                    flag = 1
                    return True, Mname
            if flag == 0:
                return False, 'NO MODEL'
    else:
        os.makedirs(MP)
        return False, 'NO MODEL'



# 尝试用，未使用
def checkModel_pick_no2(MP):
    if os.path.exists(MP):
        model_path = os.listdir(MP)
        if model_path is False:
            return False, 'NO MODEL'
        else:
            model_path.sort(key=lambda fn: os.path.getmtime(MP + '/' + fn),
                            reverse=True)
            flag = 0
            for m in model_path[3:]:
                if m.endswith('hdf5'):
                    Mname = os.path.join(MP, m)
                    flag = 1
                    return True, Mname
            if flag == 0:
                return False, 'NO MODEL'
    else:
        os.makedirs(MP)
        return False, 'NO MODEL'



def clearModel(MP):
    if os.path.exists(MP):
        model_path = os.listdir(MP)
        if model_path:
            model_path.sort(key=lambda fn: os.path.getmtime(MP + '/' + fn),
                            reverse=True)
            flag = 0
            Mname = []
            for m in model_path:
                if m.endswith('hdf5'):
                    flag += 1
                    Mname.append(os.path.join(MP, m))
            if flag > 3:
                for m in Mname[3:]:
                    os.remove(m)